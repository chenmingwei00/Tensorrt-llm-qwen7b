import argparse
import copy
import json
import os

import numpy as np
import torch
from datasets import load_dataset, load_metric
# from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm.logger import logger
# from evaluate import load
from build import get_engine_name  # isort:skip
from run import make_context, process_response
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_7b_chat.modeling_qwen import QWenLMHeadModel
from transformers.generation import GenerationConfig
from qwen_7b_chat.configuration_qwen import QWenConfig


def TRTGPT(args, config):
    dtype = config['builder_config']['precision']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    world_size = config['builder_config']['tensor_parallel']
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    use_gpt_attention_plugin = bool(
        config['plugin_config']['gpt_attention_plugin'])
    remove_input_padding = config['plugin_config']['remove_input_padding']
    multi_query_mode = config['builder_config']['multi_query_mode']
    paged_kv_cache = config['builder_config']['paged_kv_cache']

    model_config = tensorrt_llm.runtime.ModelConfig(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=hidden_size,
        gpt_attention_plugin=use_gpt_attention_plugin,
        multi_query_mode=multi_query_mode,
        remove_input_padding=remove_input_padding,
        paged_kv_cache=paged_kv_cache)

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    engine_name = get_engine_name('gpt', dtype, world_size, runtime_rank)
    serialize_path = os.path.join(args.engine_dir, engine_name)

    tensorrt_llm.logger.set_level(args.log_level)

    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping)

    return decoder


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    test_hf = args.test_hf and runtime_rank == 0  # only run hf on rank 0
    test_trt_llm = args.test_trt_llm
    hf_model_location = args.hf_model_location

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir, trust_remote_code=True, resume_download=True,
    )

    tokenizer.pad_token = tokenizer.im_end_id

    if args.eval_type == 'code_completion':
        dataset_name = "openai_humaneval"
        dataset_revision = None
        dataset_input_key = 'prompt'
        dataset_output_key = 'canonical_solution'
    elif args.eval_type == 'summarize':
        dataset_name = "ccdv/cnn_dailymail"
        dataset_revision = "3.0.0"
        dataset_input_key = 'article'
        dataset_output_key = 'highlights'
    if args.use_download_cnn:
        dataset = load_dataset(dataset_name,
                               dataset_revision,
                               cache_dir=args.dataset_path)
    else:
        import pickle
        cahe_data_path='/root/.cache/huggingface/datasets/cnn_dailymail/3.0.0/3.0.0/0107f7388b5c6fae455a5661bcd134fc22da53ea75852027040d8d1e997f101f/'
        if os.path.exists(cahe_data_path)==False:
            print('11111')
            os.makedirs(cahe_data_path)
        os.system("cp {}/* {}".format(args.dataset_path,cahe_data_path))
        files=open('./datasets/dataset.pkl','rb')
        dataset=pickle.load(files)
    #  split='test')
    # dataset = load_dataset(
    #                        dataset_name,
    #                        dataset_revision,
    #                        data_files=args.dataset_path)
    #    cache_dir=args.dataset_path)

    config_path = os.path.join(args.engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    max_batch_size = args.batch_size

    # runtime parameters
    # repetition_penalty = 1
    top_k = args.top_k
    output_len = args.output_len
    test_token_num = 923
    # top_p = 0.0
    # random_seed = 5
    temperature = 1
    num_beams = args.num_beams

    pad_id = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
    end_id = tokenizer.im_end_id

    if test_trt_llm:
        dtype = config['builder_config']['precision']
        world_size = config['builder_config']['tensor_parallel']
        num_heads = config['builder_config']['num_heads'] // world_size
        hidden_size = config['builder_config']['hidden_size'] // world_size
        vocab_size = config['builder_config']['vocab_size']
        num_layers = config['builder_config']['num_layers']
        use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)

        engine_name = get_engine_name('qwen7b', dtype, world_size, runtime_rank)
        serialize_path = os.path.join(args.engine_dir, engine_name)
        with open(serialize_path, 'rb') as f:
            engine_buffer = f.read()

        model_config = ModelConfig(model_name="qwen7b",
                                   num_heads=num_heads,
                                   hidden_size=hidden_size,
                                   vocab_size=vocab_size,
                                   num_layers=num_layers,
                                   gpt_attention_plugin=use_gpt_attention_plugin)
        decoder = tensorrt_llm.runtime.ChatGLM6BHeadModelGenerationSession(
            model_config, engine_buffer, runtime_mapping, debug_mode=False)

        system = 'You are a helpful assistant.'
        sampling_config = SamplingConfig(end_id=tokenizer.im_end_id, pad_id=0)

    if test_hf:
        model = QWenLMHeadModel.from_pretrained(hf_model_location, device_map="auto", trust_remote_code=True,
                                                fp16=True).eval()

        model.generation_config = GenerationConfig.from_pretrained(
            hf_model_location, trust_remote_code=True, resume_download=True,
        )
        model.cuda()
        # if args.data_type == 'fp16':
        #     model.half()

    def eval_tensorrt_llm(input_text, system, sampling_config):
        # the same to qwen7B
        _, input_ids = make_context(
            tokenizer,
            input_text,
            history=[],
            system=system,
            max_window_size=6144,
            chat_format='chatml',
        )
        input_ids = torch.Tensor([input_ids]).int().contiguous().cuda()
        input_lengths = torch.tensor(
            [input_ids.size(1) for _ in range(input_ids.size(0))]).int().cuda()

        decoder.setup(input_ids.size(0), input_ids.size(1), args.max_output_len)
        output_ids = decoder.decode(input_ids, input_lengths, sampling_config)
        torch.cuda.synchronize()

        for i in range(len(output_ids.tolist())):
            output_beams_list = [
                tokenizer.batch_decode(output_ids[batch_idx, :,
                                       input_lengths[batch_idx]:],
                                       skip_special_tokens=True)
                for batch_idx in range(input_lengths.size(0))
            ]
            output_text = process_response(output_beams_list[0])
        return output_text

    def eval_hf(datapoint, eval_type='summarize'):

        response = model.chat(tokenizer, datapoint, history=[])

        return [response[0]]

    if test_trt_llm:
        datapoint = dataset['test'][0:1]
        articel = datapoint[dataset_input_key][0] + 'What is the summary of this sentence?'

        output = eval_tensorrt_llm(articel,
                                   system,
                                   sampling_config)
        if runtime_rank == 0:
            logger.info(
                "---------------------------------------------------------")
            logger.info("TensorRT-LLM Generated : ")
            logger.info(f" Input : {datapoint[dataset_input_key]}")
            logger.info(f"\n Reference : {datapoint[dataset_output_key]}")
            logger.info(f"\n Output : {output}")
            logger.info(
                "---------------------------------------------------------")

    if test_hf:
        datapoint = dataset['test'][0:1]
        articel = datapoint[dataset_input_key][0] + 'What is the summary of this sentence?'
        output = eval_hf(articel,
                         eval_type=args.eval_type)
        logger.info("---------------------------------------------------------")
        logger.info("HF Generated : ")
        logger.info(f" Input : {datapoint[dataset_input_key]}")
        logger.info(f"\n Reference : {datapoint[dataset_output_key]}")
        logger.info(f"\n Output : {output}")
        logger.info("---------------------------------------------------------")
    if args.use_download_cnn:
        metric_tensorrt_llm = [load_metric("rouge") for _ in range(num_beams)]
        metric_hf = [load_metric("rouge") for _ in range(num_beams)]
    else:
        metric_tensorrt_llm = [load_metric(args.rouge_path) for _ in range(num_beams)]
        metric_hf = [load_metric(args.rouge_path) for _ in range(num_beams)]
    for i in range(num_beams):
        metric_tensorrt_llm[i].seed = 0
        metric_hf[i].seed = 0

    ite_count = 0
    data_point_idx = 0
    step = 0
    while (data_point_idx < len(dataset['test'])) and (ite_count <
                                                       args.max_ite):

        if runtime_rank == 0:
            logger.debug(
                f"run data_point {data_point_idx} ~ {data_point_idx + max_batch_size}"
            )
        datapoint = dataset['test'][data_point_idx:(data_point_idx +
                                                    max_batch_size)]

        if test_trt_llm:
            profiler.start('tensorrt_llm')
            if len(datapoint[dataset_input_key][0]) > args.max_input_len:
                articel = datapoint[dataset_input_key][0][:args.max_input_len]
            articel = articel + 'What is the summary of this sentence?'

            output_tensorrt_llm = eval_tensorrt_llm(articel,
                                                    system,
                                                    sampling_config)
            profiler.stop('tensorrt_llm')

        if test_hf:
            profiler.start('hf')
            if len(datapoint[dataset_input_key][0]) > args.max_input_len:
                articel = datapoint[dataset_input_key][0][:args.max_input_len]
            articel = articel + 'What is the summary of this sentence?'
            output_hf = eval_hf(articel, )
            profiler.stop('hf')

        if runtime_rank == 0:
            if test_trt_llm:
                for batch_idx in range(len(output_tensorrt_llm)):
                    for beam_idx in range(num_beams):
                        metric_tensorrt_llm[beam_idx].add_batch(
                            predictions=output_tensorrt_llm,
                            references=[
                                datapoint[dataset_output_key][batch_idx]
                            ])
            if test_hf:
                for beam_idx in range(num_beams):
                    for batch_idx in range(len(output_hf)):
                        metric_hf[beam_idx].add_batch(
                            predictions=output_hf,
                            references=[
                                datapoint[dataset_output_key][batch_idx]
                            ])

            logger.debug('-' * 100)
            logger.debug(f"Input : {datapoint[dataset_input_key]}")
            if test_trt_llm:
                logger.debug(f'TensorRT-LLM Output: {output_tensorrt_llm}')
            if test_hf:
                logger.debug(f'HF Output: {output_hf}')
            logger.debug(f"highlights : {datapoint[dataset_output_key]}")

        data_point_idx += max_batch_size
        ite_count += 1
        # step+=1

    if runtime_rank == 0:
        if test_trt_llm:
            np.random.seed(0)  # rouge score use sampling to compute the score
            logger.info(
                f'TensorRT-LLM (total latency: {profiler.elapsed_time_in_sec("tensorrt_llm")} sec)'
            )
            for beam_idx in range(num_beams):
                logger.info(f"TensorRT-LLM beam {beam_idx} result")
                computed_metrics_tensorrt_llm = metric_tensorrt_llm[
                    beam_idx].compute()
                for key in computed_metrics_tensorrt_llm.keys():
                    logger.info(
                        f'  {key} : {computed_metrics_tensorrt_llm[key].mid[2] * 100}'
                    )

                if args.check_accuracy and beam_idx == 0:
                    assert computed_metrics_tensorrt_llm['rouge1'].mid[
                               2] * 100 > args.tensorrt_llm_rouge1_threshold
        if test_hf:
            np.random.seed(0)  # rouge score use sampling to compute the score
            logger.info(
                f'Hugging Face (total latency: {profiler.elapsed_time_in_sec("hf")} sec)'
            )
            for beam_idx in range(num_beams):
                logger.info(f"HF beam {beam_idx} result")
                computed_metrics_hf = metric_hf[beam_idx].compute()
                for key in computed_metrics_hf.keys():
                    logger.info(
                        f'  {key} : {computed_metrics_hf[key].mid[2] * 100}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_location', type=str, default='/root/workspace/trt2023/QWen-7B-Chat/')
    parser.add_argument(
        '--tokenizer',
        default=None,
        help='tokenizer path; defaults to hf_model_location if left unspecified'
    )
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default='/root/workspace/QWen-7B-Chat',
                        help='Directory containing the tokenizer model.')
    parser.add_argument('--test_hf',default=False,action='store_true')
    parser.add_argument('--test_trt_llm',default=False,action='store_true')
    parser.add_argument('--use_download_cnn',default=False,action='store_true')

    parser.add_argument('--data_type',
                        type=str,
                        choices=['fp32', 'fp16'],
                        default='fp32')
    parser.add_argument('--rouge_path', type=str, default='./datasets/cnn_dailymail.py')

    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--engine_dir', type=str, default='/root/workspace/tensorrt_llm_july-release-v1/qwenb_chen/qwen_trtModel')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_ite', type=int, default=20)
    parser.add_argument('--output_len', type=int, default=100)
    parser.add_argument('--max_output_len', type=int, default=1024)
    parser.add_argument('--max_input_len', type=int, default=1950)

    parser.add_argument('--check_accuracy', default=True, action='store_true')
    parser.add_argument('--tensorrt_llm_rouge1_threshold',
                        type=float,
                        default=14.0)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--eval_type',
                        type=str,
                        default='summarize',
                        choices=['summarize', 'code_completion'])

    args = parser.parse_args()
    if args.tokenizer == None:
        args.tokenizer = args.hf_model_location
    main(args)
