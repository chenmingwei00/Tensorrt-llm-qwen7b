import unittest
import argparse

import numpy as np
import torch
from polygraphy.backend.trt import EngineFromNetwork, TrtRunner
from torch import nn
from pathlib import Path
from tensorrt_llm.logger import logger
import tensorrt as trt
import tensorrt_llm
from tensorrt_llm.layers import PositionEmbeddingType
from weight import load_from_ft, parse_ft_config 
from tensorrt_llm import Module, Tensor
from tensorrt_llm.quantization import QuantMode
import os
import time
import torch.multiprocessing as mp
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.models import smooth_quantize, weight_only_quantize
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.functional import RaggedTensor
MODEL_NAME = "qwen7b"

def get_engine_name(model, dtype, tp_size, rank):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')
    parser.add_argument('--model_dir', type=str, default='./qwenftModel/1-gpu')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32'])
    parser.add_argument(
        '--timing_cache',
        type=str,
        default='model.cache',
        help=
        'The path of to read timing cache from, will be ignored if the file does not exist'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='verbose',
        choices=['verbose', 'info', 'warning', 'error', 'internal_error'])
    parser.add_argument('--vocab_size', type=int, default=130528)
    parser.add_argument('--n_layer', type=int, default=32)
    parser.add_argument('--n_positions', type=int, default=8192)
    parser.add_argument('--n_embd', type=int, default=4096)
    parser.add_argument('--n_head', type=int, default=32)
    parser.add_argument('--hidden_act', type=str, default='silu') #gelu
    parser.add_argument(
        '--rotary_pct',
        type=float,
        default=0.0,
        help="Setting this to a value > 0.0 (and <= 1.0) activates RoPE.")
    parser.add_argument('--inter_size', type=int, default=None)
    parser.add_argument('--no_bias', action="store_false")
    parser.add_argument('--max_batch_size', type=int, default=1)
    parser.add_argument('--max_input_len', type=int, default=1024)
    parser.add_argument('--max_output_len', type=int, default=1024)
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument('--use_gpt_attention_plugin',
                        nargs='?',
                        const='float16',
                        default='float16',
                        choices=['float16', 'float32', False])
    parser.add_argument('--use_gemm_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16', 'float32'])
    parser.add_argument('--use_layernorm_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16', 'float32'])
    parser.add_argument('--parallel_build', default=False, action='store_true')
    parser.add_argument('--enable_context_fmha',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_context_fmha_fp32_acc',
                        default=False,
                        action='store_true')
    parser.add_argument('--gpus_per_node', type=int, default=8)
    parser.add_argument('--builder_opt', type=int, default=None)
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./qwen_trtModel',
        help=
        'The path to save the serialized engine files, timing cache file and model configs'
    )
    parser.add_argument(
        "--multi_query_mode",
        "-mq",
        default=False,
        action='store_true',
        help=
        "Whether this model uses multi-query attention mechanism (default: False)"
    )
    # Arguments related to the quantization of the model.
    parser.add_argument(
        '--use_smooth_quant',
        default=False,
        action="store_true",
        help=
        'Use the SmoothQuant method to quantize activations and weights for the various GEMMs.'
        'See --per_channel and --per_token for finer-grained quantization options.'
    )
    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision')
    parser.add_argument(
        '--weight_only_precision',
        const='int8',
        type=str,
        nargs='?',
        default='int8',
        choices=['int8', 'int4'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    parser.add_argument(
        '--per_channel',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor for the GEMM\'s result. '
        'per_channel instead uses a different static scaling factor for each channel. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        '--per_token',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor to scale activations in the int8 range. '
        'per_token chooses at run time, and for each token, a custom scaling factor. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        '--int8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=None,
        help=
        'Seed to use when initializing the random number generator for torch.')

    args = parser.parse_args()
    assert not (
        args.use_smooth_quant and args.use_weight_only
    ), "You cannot enable both SmoothQuant and INT8 weight-only together."

    if args.use_smooth_quant:
        args.quant_mode = QuantMode.use_smooth_quant(args.per_token,
                                                     args.per_channel)
    elif args.use_weight_only:
        args.quant_mode = QuantMode.use_weight_only(
            args.weight_only_precision == 'int4')
    else:
        args.quant_mode = QuantMode(0)
    args.bias = not args.no_bias

    if args.inter_size is None:
        args.inter_size = 4 * args.n_embd

    if args.int8_kv_cache:
        assert (
            args.use_gpt_attention_plugin
        ), "You have to use GPT attention plugin when int8 KV cache is set"
        args.quant_mode = args.quant_mode.set_int8_kv_cache()

    if args.model_dir is not None:
        n_embd, n_head, n_layer, n_positions, vocab_size, _, hidden_act, rotary_pct, bias, inter_size, multi_query_mode,\
            layer_norm_epsilon,kv_channels = parse_ft_config(
            Path(args.model_dir) / "config.ini")
        args.hidden_act='silu'
        args.n_embd = n_embd
        args.n_head = n_head
        args.n_layer = n_layer
        args.n_positions = 8192
        args.vocab_size = vocab_size
        args.hidden_act = hidden_act
        args.rotary_pct = rotary_pct
        args.bias = bias
        args.inter_size = inter_size
        args.multi_query_mode = multi_query_mode
        args.layer_norm_epsilon = layer_norm_epsilon
        args.kv_channels = kv_channels

    return args

class TestDebuggingAPI(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def test_debugging_api(self,args):
        rank=0
        cur_rank=0
        args.hidden_act='silu'

        torch.cuda.set_device(rank % args.gpus_per_node)
        tensorrt_llm.logger.set_level(args.log_level)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        engine_name='./qwen7b_float16_tp1_rank0.engine'
        # test data
        dtype = 'float16'
       
        # construct trt network
        apply_query_key_layer_scaling = False
        builder = tensorrt_llm.Builder()
        cache = None

        builder_config = builder.create_builder_config(
            name=MODEL_NAME,
            precision=args.dtype,
            timing_cache=args.timing_cache if cache is None else cache,
            tensor_parallel=args.world_size,  # TP only
            parallel_build=args.parallel_build,
            num_layers= args.n_layer,
            num_heads=args.n_head,
            hidden_size=args.n_embd,
            vocab_size=args.vocab_size,
            hidden_act=args.hidden_act,
            max_position_embeddings=args.n_positions,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            int8=(args.quant_mode.has_act_and_weight_quant()
                  or args.quant_mode.has_int8_kv_cache()),
            opt_level=args.builder_opt,
            multi_query_mode=args.multi_query_mode)
        
        engine_name = get_engine_name(MODEL_NAME, args.dtype, args.world_size,
                                      0)
        
        kv_dtype =tensorrt_llm.str_dtype_to_trt(args.dtype)
            # builder_config=tensorrt_llm.builder.BuilderConfig
           
            # Initialize Module Qwen7BHeadModel
           
        tensorrt_llm_Qwen7BModel = tensorrt_llm.models.Qwen7BHeadModel(
                                                            num_layers=args.n_layer,
                                                            num_heads=args.n_head,
                                                            hidden_size=args.n_embd,
                                                            inter_size=args.inter_size,
                                                            vocab_size=args.vocab_size,
                                                            hidden_act=args.hidden_act,
                                                            max_position_embeddings=args.n_positions,
                                                            position_embedding_type=PositionEmbeddingType.learned_absolute
                                                            if args.rotary_pct == 0.0 else PositionEmbeddingType.rope,
                                                            rotary_embedding_percentage=args.rotary_pct,
                                                            dtype=kv_dtype,
                                                            tensor_parallel=args.world_size,  # TP only
                                                            tensor_parallel_group=list(range(args.world_size)),  # TP only
                                                            apply_query_key_layer_scaling=builder_config.
                                                            apply_query_key_layer_scaling,
                                                            quant_mode=args.quant_mode,
                                                            bias=args.bias,
                                                            multi_query_mode=args.multi_query_mode,
                                                            layer_norm_epsilon=args.layer_norm_epsilon,
                                                            kv_channels=args.kv_channels)
        
        if args.use_smooth_quant:
            tensorrt_llm_Qwen7BModel = smooth_quantize(tensorrt_llm_Qwen7BModel, args.quant_mode)
        elif args.use_weight_only:
            tensorrt_llm_Qwen7BModel = weight_only_quantize(tensorrt_llm_Qwen7BModel, args.quant_mode)
            
        if args.model_dir is not None:
            load_from_ft(tensorrt_llm_Qwen7BModel,
                            args.model_dir,
                            rank,
                            args.world_size,
                            fp16=(args.dtype == 'float16'))
            
        network = builder.create_network()

        network.trt_network.name = engine_name
        if args.use_gpt_attention_plugin:
            network.plugin_config.set_gpt_attention_plugin(
                dtype=args.use_gpt_attention_plugin)
        
        if args.use_gemm_plugin:
            network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
        if args.use_layernorm_plugin:
            network.plugin_config.set_layernorm_plugin(
                                dtype=args.use_layernorm_plugin)
        assert not (args.enable_context_fmha and args.enable_context_fmha_fp32_acc)
        if args.enable_context_fmha:
            network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
        if args.enable_context_fmha_fp32_acc:
            network.plugin_config.set_context_fmha(
                                ContextFMHAType.enabled_with_fp32_acc)
        

        # Quantization plugins.
        if args.use_smooth_quant:
            network.plugin_config.set_smooth_quant_gemm_plugin(dtype=args.dtype)
            network.plugin_config.set_layernorm_quantization_plugin(
                dtype=args.dtype)
            # FIXME(nkorobov)
            # See https://nvbugs/4164762
            # See https://nvbugs/4174113
            network.plugin_config.set_quantize_tensor_plugin()
            network.plugin_config.set_quantize_per_token_plugin()
        elif args.use_weight_only:
            network.plugin_config.set_weight_only_quant_matmul_plugin(
                dtype='float16')

        if args.world_size > 1:
            network.plugin_config.set_nccl_plugin(args.dtype)
        
        with net_guard(network):
             # Prepare
            network.set_named_parameters(
                tensorrt_llm_Qwen7BModel.named_parameters())
            
            inputs = tensorrt_llm_Qwen7BModel.prepare_inputs(
                    args.max_batch_size, args.max_input_len, args.max_output_len, True,
                    args.max_beam_width)
            output=tensorrt_llm_Qwen7BModel(*inputs)
            
            # network._mark_output(output[0], 'output',
            #                  tensorrt_llm.str_dtype_to_trt(dtype))
            for k, v in tensorrt_llm_Qwen7BModel.named_network_outputs():
                network._mark_output(v, k, tensorrt_llm.str_dtype_to_trt(dtype))
            

        #     # for debug models construct same input about qwen7b forward
        #     import pickle
        #     with open('/root/workspace/trt2023/input.pickle', 'rb' ) as file_name:
        #         input =pickle.load(file_name)

        #     input_ids_data=input[0]
        #     position_idss_data=np.transpose(np.array([input[2],input[2]]),(1, 0, 2))
        #     masked_tokenss_data=input[1]
        #     last_token_ids_data=np.array([input_ids_data.shape[1]])

        engine = None

        # Network -> Engine
        engine = builder.build_engine(network, builder_config)
        if rank == 0:
            config_path = os.path.join(args.output_dir, 'config.json')
            builder.save_config(builder_config, config_path)

        assert engine is not None, f'Failed to build engine for rank {cur_rank}'

        if cur_rank == 0:
            # Use in-memory timing cache for multiple builder passes.
            if not args.parallel_build:
                cache = builder_config.trt_builder_config.get_timing_cache()

        serialize_engine(engine, os.path.join(args.output_dir, engine_name))
     

if __name__ == '__main__':
    args = parse_arguments()
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)

    logger.set_level(args.log_level)

    debug_mlp=TestDebuggingAPI()
    if args.parallel_build and args.world_size > 1 and \
            torch.cuda.device_count() >= args.world_size:
        logger.warning(
            f'Parallelly build TensorRT engines. Please make sure that all of the {args.world_size} GPUs are totally free.'
        )
        mp.spawn(build, nprocs=args.world_size, args=(args, ))
    else:
        args.parallel_build = False
        logger.info('Serially build TensorRT engines.')
        # build(0, args)
        debug_mlp.test_debugging_api(args)