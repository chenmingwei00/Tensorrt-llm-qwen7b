import math
from collections import OrderedDict

import numpy as np
import tensorrt as trt

from ..._common import default_net,default_trtnet, precision
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import (RaggedTensor, Tensor, assertion, concat, constant,
                           expand, gather, gpt_attention, shape, split,_create_tensor,pow,cast)
from ...layers import (MLP, AttentionMaskType, ColumnLinear, Embedding,
                       LayerNorm, PositionEmbeddingType, RowLinear,RmsNorm)
from ...module import Module, ModuleList
from ...parameter import Parameter
from ...quantization import QuantMode
from einops import rearrange

from ...plugin import  _TRT_LLM_PLUGIN_NAMESPACE as TRT_LLM_PLUGIN_NAMESPACE
print('333333333333333333333333333333333333')

def identity_op(tensor: Tensor) -> Tensor:
    input_tensor = tensor.trt_tensor
    # Create a plugin instance.
    plugin_creator = trt.get_plugin_registry().get_plugin_creator(
        'Identity', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plugin_creator is not None
    pfc = trt.PluginFieldCollection([])
    plugin = plugin_creator.create_plugin("identity", pfc)
    layer = default_trtnet().add_plugin_v2([input_tensor], plugin)
    return _create_tensor(layer.get_output(0), layer)
class QWenAttention(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 num_layers=1,
                 apply_query_key_layer_scaling=False,
                 attention_mask_type=AttentionMaskType.bidirectional,
                 bias=True,
                 dtype=None,
                 position_embedding_type=PositionEmbeddingType.learned_absolute,
                 neox_rotary_style=False,
                 use_int8_kv_cache=False,
                 rotary_embedding_percentage=1.0,
                 tp_group=None,
                 tp_size=1,
                 multi_block_mode=False,
                 multi_query_mode=False,
                 kv_channels=128):
        super().__init__()

        self.attention_mask_type = attention_mask_type
        self.num_heads = num_attention_heads // tp_size
        self.attention_head_size =  int(hidden_size/ self.num_heads)
        self.num_attention_kv_heads = 1 if multi_query_mode else self.num_heads
        self.hidden_size = hidden_size // tp_size
        self.max_positions = max_position_embeddings
        self.num_layers = num_layers
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.q_scaling = 1
        if self.apply_query_key_layer_scaling:
            self.norm_factor *= self.num_layers
            self.q_scaling *= self.num_layers

        self.position_embedding_type = position_embedding_type
        self.multi_block_mode = multi_block_mode
        self.multi_query_mode = multi_query_mode

        self.rotary_embedding_dim = 0
        self.neox_rotary_style = neox_rotary_style
        # if self.position_embedding_type == PositionEmbeddingType.rope:
        #     self.rotary_embedding_dim = int(self.attention_head_size *
        #                                     rotary_embedding_percentage)
        #     # TODO: Once we add RotaryEmbedding outside GPTAttention plugin,
        #     #       we need to set it up here

        self.dtype = dtype
       
        # modfiy by qwen7B
        self.split_size = hidden_size
        self.head_dim = hidden_size // num_attention_heads
        self.projection_size = kv_channels * num_attention_heads
        self.hidden_size_per_attention_head = (
         self.projection_size // num_attention_heads)

        self.num_attention_heads=num_attention_heads
        self.use_int8_kv_cache = use_int8_kv_cache
        if self.use_int8_kv_cache:
            self.kv_orig_quant_scale = Parameter(shape=(1, ), dtype='float32')
            self.kv_quant_orig_scale = Parameter(shape=(1, ), dtype='float32')
        else:
            self.register_parameter('kv_orig_quant_scale', None)
            self.register_parameter('kv_quant_orig_scale', None)

        # Note: in multi_query_mode, only query heads are split between multiple GPUs,
        # while key/value head are not split as there is only one head per key/value.
        # The output feature size is therefore (h/tp + 2) * d, where h is num_heads,
        # d is head_size, and tp is tensor_parallel_size.
        # if not multi_query_mode else hidden_size +
                                    # 2 * tp_size * self.attention_head_size,
        # In ColumnLinear op, the output dim is calculated by (h + 2*tp) * d / tp,
        # which matches the desired output size (h/tp + 2) * d after splitting
        # self.c_attn = nn.Linear(config.hidden_size, 3 * self.projection_size)

        
        self.c_attn  = ColumnLinear(hidden_size,
                                    hidden_size*3, 
                                    bias=bias,
                                    dtype=dtype,
                                    tp_group=tp_group,
                                    tp_size=tp_size,
                                    gather_output=False)
        self.c_proj = RowLinear(hidden_size,
                               hidden_size,
                               bias=None,
                               dtype=dtype,
                               tp_group=tp_group,
                               tp_size=tp_size)
    
    def _split_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.view(
                            concat([
                                shape(tensor, 0),
                                shape(tensor, 1), 
                                num_heads,
                                attn_head_size
                            ]))
        # new_shape = () + (num_heads, attn_head_size)
        # tensor = tensor.view(new_shape)
        return tensor

    def forward(self,
                hidden_states: RaggedTensor,
                position_embedding,
                past_key_value,
                sequence_length,
                past_key_value_length,
                masked_tokens,
                cache_indirection,
                use_cache=False):

        if not default_net().plugin_config.gpt_attention_plugin:
            raise ValueError(
                'ChatGLM is only supported with GPTAttention plugin')

        assert isinstance(hidden_states, RaggedTensor)
        input_lengths = hidden_states.row_lengths
        max_input_length = hidden_states.max_row_length
        hidden_states = hidden_states.data

        qkv = self.c_attn(hidden_states) #pass

        # attention
        query,key,value=split(qkv,self.split_size,dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)  #pass

        zero = constant(
            np.ascontiguousarray(
                np.zeros([1, 1, 1, 1],
                         dtype=np.float16
                         if self.dtype == trt.float16 else np.float32)))

        def rotate(x64):
            x32_part0, x32_part1 = x64.split(64, dim=-1)

            x32_part1_negtive = zero - x32_part1

            y64 = concat([x32_part1_negtive, x32_part0], dim=3)
            return y64

        def rotate_embedding(x, position_embedding_value,position_embedding_value_sin): # the same to qwen7B ApplyRotaryEmb

            t_ = (x * position_embedding_value) + (rotate(x) *position_embedding_value_sin)

            return t_

        cos0, cos1,sin0,sin1 = position_embedding
      
        query = rotate_embedding(query,cos0,sin0) #the same to qwen7B
        key = rotate_embedding(key,cos1,sin1)  #pass
      
        kv_orig_quant_scale = self.kv_orig_quant_scale.value if self.use_int8_kv_cache else None
        kv_quant_orig_scale = self.kv_quant_orig_scale.value if self.use_int8_kv_cache else None

        qkv = concat([query, key, value], dim=2)
        qkv = qkv.view(
            concat([shape(qkv, 0),
                    shape(qkv, 1), self.hidden_size * 3]))

        context, past_key_value = gpt_attention(
            qkv,
            past_key_value,
            sequence_length,
            past_key_value_length,
            masked_tokens,
            input_lengths,
            max_input_length,
            cache_indirection,
            self.num_attention_heads,
            self.attention_head_size,
            self.q_scaling,
            self.rotary_embedding_dim,
            self.neox_rotary_style,
            self.multi_block_mode,
            self.multi_query_mode,
            kv_orig_quant_scale,
            kv_quant_orig_scale,
            self.use_int8_kv_cache,
            mask_type=1) #self.attention_mask_type.value

        context = self.c_proj(context)

        context = RaggedTensor.from_row_lengths(context, input_lengths,
                                                max_input_length)
        if use_cache:
            return (context, past_key_value)  #pass
        else:
            return context


class QWenBlock(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 num_layers,
                 dtype=None,
                 apply_query_key_layer_scaling=False,
                 attention_mask_type=AttentionMaskType.bidirectional,
                 hidden_act='relu',
                 position_embedding_type=PositionEmbeddingType.learned_absolute,
                 quant_mode=QuantMode(0),
                 rotary_embedding_percentage=1.0,
                 inter_size=None,
                 bias=True,
                 multi_query_mode=False,
                 tp_group=None,
                 tp_size=1,
                 layer_norm_epsilon=1e-6,
                 kv_channels=128):
        super().__init__()
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_mask_type = attention_mask_type
        self.hidden_act = hidden_act
        self.position_embedding_type = position_embedding_type
        self.tp_group = tp_group
        self.tp_size = tp_size
        # self.input_layernorm = LayerNorm(normalized_shape=hidden_size,
        #                                  dtype=dtype)
        
        self.ln_1=RmsNorm(normalized_shape=hidden_size,
                          eps=layer_norm_epsilon,
                          dtype=dtype)
        self.ln_2 = RmsNorm(
            hidden_size,
            eps=layer_norm_epsilon,
            dtype=dtype
        )

        self.attn = QWenAttention(
            hidden_size,
            num_attention_heads,
            max_position_embeddings,
            num_layers,
            apply_query_key_layer_scaling,
            dtype=dtype,
            attention_mask_type=attention_mask_type,
            position_embedding_type=position_embedding_type,
            neox_rotary_style=True,
            rotary_embedding_percentage=rotary_embedding_percentage,
            bias=bias,
            tp_group=tp_group,
            tp_size=tp_size,
            use_int8_kv_cache=quant_mode.has_int8_kv_cache(),
            kv_channels=kv_channels)

        if inter_size is None:
            inter_size = hidden_size * 4

        self.mlp = MLP(hidden_size=hidden_size,
                       ffn_hidden_size=inter_size,
                       hidden_act=hidden_act,
                       dtype=dtype,
                       bias=bias,
                       tp_group=tp_group,
                       tp_size=tp_size)
        # self.post_layernorm = LayerNorm(normalized_shape=hidden_size,
        #                                 dtype=dtype)

    def forward(self,
                hidden_states: RaggedTensor,
                position_embedding,
                past_key_value=None,
                sequence_length=None,
                past_key_value_length=None,
                masked_tokens=None,
                use_cache=False,
                cache_indirection=None):

        assert isinstance(hidden_states, RaggedTensor)
        # unpack the RaggedTensor since some layers like MLP, LayerNorm only need data tensor
        input_lengths = hidden_states.row_lengths
        max_input_length = hidden_states.max_row_length
        hidden_states = hidden_states.data

        layernorm_output = self.ln_1(hidden_states)
        attention_output = self.attn(
            RaggedTensor.from_row_lengths(layernorm_output, input_lengths,
                                          max_input_length),
            position_embedding=position_embedding,
            past_key_value=past_key_value,
            sequence_length=sequence_length,
            past_key_value_length=past_key_value_length,
            masked_tokens=masked_tokens,
            cache_indirection=cache_indirection,
            use_cache=use_cache,
        )

        if use_cache:
            attention_output, presents = attention_output


        layernorm_input = hidden_states + attention_output.data  # chatglm7B is: hidden_states * 7.484375 + attention_output.data

        hidden_states = self.ln_2(layernorm_input) #pass

        mlp_output = self.mlp(hidden_states) #pass

        hidden_states = layernorm_input + mlp_output  #chatglm7B is: hidden_states * 7.484375 + mlp_output pass

        hidden_states = RaggedTensor.from_row_lengths(
            hidden_states, attention_output.row_lengths, max_input_length)

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class Qwen7BModel(Module):

    def __init__(self,
                 num_layers, #1
                 num_heads,
                 hidden_size, #1
                 vocab_size, # 1
                 hidden_act,
                 max_position_embeddings,
                 dtype=None,
                 tensor_parallel=1,
                 tensor_parallel_group=None,
                 apply_query_key_layer_scaling=False,
                 position_embedding_type=PositionEmbeddingType.learned_absolute,
                 rotary_embedding_percentage=1.0,
                 inter_size=None,
                 bias=True,
                 quant_mode=QuantMode(0),
                 multi_query_mode=False,
                 gradient_checkpointing=False,
                 use_dynamic_ntk=True,
                 seq_length=2048,
                 rotary_ndims=None,
                 layer_norm_epsilon=1e-6,
                 kv_channels=128):
        super().__init__()

        self.half_head_size = hidden_size // num_heads // 2    #64 

        self.wte = Embedding(vocab_size, hidden_size, dtype=dtype)
        #以下对应RotaryEmbedding
        self.position_embedding_cos = Embedding(max_position_embeddings,
                                                self.half_head_size*2,
                                                dtype=dtype)  # shape=8192,128
        self.position_embedding_sin = Embedding(max_position_embeddings,
                                                self.half_head_size*2,
                                                dtype=dtype)

        self.h = ModuleList([
            QWenBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                max_position_embeddings=max_position_embeddings,
                num_layers=num_layers,
                dtype=dtype,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                attention_mask_type=AttentionMaskType.bidirectional,
                hidden_act=hidden_act,
                position_embedding_type=position_embedding_type,
                rotary_embedding_percentage=rotary_embedding_percentage,
                multi_query_mode=multi_query_mode,
                tp_group=tensor_parallel_group,
                tp_size=tensor_parallel,
                inter_size=inter_size,
                bias=bias,
                quant_mode=quant_mode,
                layer_norm_epsilon=layer_norm_epsilon,
                kv_channels=kv_channels) for _ in range(num_layers)
        ])

        self.ln_f = RmsNorm_my(normalized_shape=hidden_size,
                                eps=layer_norm_epsilon,
                                dtype=dtype)
        
        # rmsnorm
        if isinstance(hidden_size, int):
            normalized_shape = (hidden_size, )
        self.normalized_shape = tuple(normalized_shape)

        # self.lf_weight= Parameter(shape=self.normalized_shape, dtype=dtype)
        # self.dtype=dtype
        # self.eps=layer_norm_epsilon

    def forward(self,
                input_ids=None,
                position_ids=None,
                past_key_value=None,
                sequence_length=None,
                past_key_value_length=None,
                masked_tokens=None,
                use_cache=False,
                cache_indirection=None):

        batch_size = shape(input_ids.data, 0)
        input_len = shape(input_ids.data, 1)

        hidden_states = self.wte(input_ids.data)
        position_embedding_cos = self.position_embedding_cos(position_ids) #  Previously calculated 
        position_embedding_sin = self.position_embedding_sin(position_ids) #  Previously calculated 
        # position_embedding_cos=unsqueeze(position_embedding_cos,  2)
        # position_embedding_sin=unsqueeze(position_embedding_sin,  2)

        # rotary_pos_emb = ([position_embedding_cos,position_embedding_sin],) * 2
        # q_pos_emb, k_pos_emb = rotary_pos_emb


        position_embedding_cos0, position_embedding_cos1 = position_embedding_cos.split(
            1, dim=1)
        position_embedding_sin0, position_embedding_sin1 = position_embedding_sin.split(
            1, dim=1)

      
        position_embedding_cos0 = position_embedding_cos0.view(
            concat([batch_size, input_len, 1, self.half_head_size*2]))
        position_embedding_cos1 = position_embedding_cos1.view(
            concat([batch_size, input_len, 1, self.half_head_size*2]))
        position_embedding_sin0 = position_embedding_sin0.view(
            concat([batch_size, input_len, 1, self.half_head_size*2]))
        position_embedding_sin1 = position_embedding_sin1.view(
            concat([batch_size, input_len, 1, self.half_head_size*2]))

        position_embedding = [
            position_embedding_cos0, position_embedding_cos1,
            position_embedding_sin0, position_embedding_sin1
        ]

        if past_key_value is None:
            past_key_value = tuple([None] * len(self.h))

        if use_cache:
            presents = []

        hidden_states = RaggedTensor.from_row_lengths(hidden_states,
                                                      input_ids.row_lengths,
                                                      input_ids.max_row_length)
        for layer, past in zip(self.h, past_key_value):
            hidden_states = layer(hidden_states,
                                  position_embedding,
                                  past_key_value=past,
                                  sequence_length=sequence_length,
                                  past_key_value_length=past_key_value_length,
                                  masked_tokens=masked_tokens,
                                  use_cache=use_cache,
                                  cache_indirection=cache_indirection)

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]
        hidden_states=hidden_states.data

        # self.register_network_output('hidden_states0', identity_op(hidden_states))
        hidden_states_clone=identity_op(hidden_states)
        hidden_states_final = self.ln_f(hidden_states_clone)
        # self.register_network_output('hidden_states_final', identity_op(hidden_states_final))

        if use_cache:
            return (hidden_states_final, tuple(presents))
        return hidden_states_final


class Qwen7BHeadModel(Qwen7BModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.rotary_emb\.inv_freq"]
    _keys_to_ignore_on_load_unexpected = [r"h\.\d+\.attn\.masked_bias"]

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 dtype,
                 tensor_parallel=1,
                 tensor_parallel_group=None,
                 apply_query_key_layer_scaling=False,
                 position_embedding_type=PositionEmbeddingType.learned_absolute,
                 rotary_embedding_percentage=1.0,
                 inter_size=None,
                 bias=True,
                 quant_mode=QuantMode(0),
                 multi_query_mode=False,
                 layer_norm_epsilon=1e-6,
                 kv_channels=128):
              
        if isinstance(dtype, str):
            self._kv_dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self._kv_dtype = dtype
        self._dtype = self._kv_dtype
        if quant_mode.has_int8_kv_cache():
            self._kv_dtype = str_dtype_to_trt('int8')

        self.quant_mode = quant_mode

        self._num_layers = num_layers
        self._num_heads = num_heads
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._tensor_parallel = tensor_parallel
        self._multi_query_mode = multi_query_mode
        super().__init__(num_layers, num_heads, hidden_size, vocab_size,
                         hidden_act, max_position_embeddings, dtype,
                         tensor_parallel, tensor_parallel_group,
                         apply_query_key_layer_scaling, position_embedding_type,
                         rotary_embedding_percentage, inter_size, bias,
                         quant_mode, multi_query_mode,layer_norm_epsilon=layer_norm_epsilon,
                         kv_channels=kv_channels)
        vocab_size_padded = pad_vocab_size(vocab_size, tensor_parallel)
        self.lm_head = ColumnLinear(hidden_size,
                                    vocab_size_padded,
                                    bias=False,
                                    dtype=dtype,
                                    tp_group=tensor_parallel_group,
                                    tp_size=tensor_parallel,
                                    gather_output=True)

    def forward(self,
                input_ids=None,
                position_ids=None,
                past_key_value=None,
                sequence_length=None,
                past_key_value_length=None,
                masked_tokens=None,
                use_cache=False,
                last_token_ids=None,
                cache_indirection=None):
        assert last_token_ids is not None, "Expecting last token ids to be not None"

        hidden_states = super().forward(input_ids, position_ids, past_key_value,
                                        sequence_length, past_key_value_length,
                                        masked_tokens, use_cache,
                                        cache_indirection)

        if use_cache:
            hidden_states, presents = hidden_states

        # only calculate logits for the last token
        # [batch_size, seqlen, hidden_size] -> [batch_size, hidden_size]
        last_token_ids = last_token_ids.view(
            concat([shape(last_token_ids, 0), 1, 1]))
        last_token_ids = expand(
            last_token_ids,
            concat([shape(last_token_ids, 0), 1,
                    shape(hidden_states, 2)]))
        last_token_ids = last_token_ids - 1
        hidden_states = gather(
            hidden_states, dim=1, indices=last_token_ids).view(
                concat([shape(hidden_states, 0),
                        shape(hidden_states, 2)]))

        # [batch_size, hidden_size] -> [batch_size, vocab_size]
        lm_logits = self.lm_head(hidden_states)
        lm_logits =identity_op(lm_logits)
        lm_logits.mark_output('logits', self._dtype)
        # self.register_network_output('lm_logits', lm_logits)

        # out_inter.mark_output('inter', str_dtype_to_trt('float32'))

        if use_cache:
            for i, present in enumerate(presents):
                present.mark_output(f'present_key_value_{i}', self._kv_dtype)
            return (lm_logits, presents)

        return lm_logits

    def prepare_inputs(self,
                       max_batch_size,
                       max_input_len,
                       max_new_tokens,
                       use_cache,
                       max_beam_width: int = 1):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        head_size = self._hidden_size // self._num_heads
        num_heads = self._num_heads // self._tensor_parallel
        num_heads_kv = 1 if self._multi_query_mode else num_heads
        max_len = max_input_len + max_new_tokens 
        bb_range = [
            1, (max_batch_size * max_beam_width + 1) // 2,
            max_batch_size * max_beam_width
        ]
        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        beam_width_range = [1, (max_beam_width + 1) // 2, max_beam_width]
        inlen_range = [1, 1, max_input_len]
        max_len_range = [0, (max_len + 1) // 2, max_len]
        [1, (max_len + 1) // 2 + 1, max_len + 1]

        past_key_value = []
        sequence_length = None
        past_key_value_length = None
        masked_tokens = None
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin

        input_ids = Tensor(name='input_ids',
                           dtype=trt.int32,
                           shape=[-1, -1],
                           dim_range=OrderedDict([
                               ('batch_size', [bb_range]), 
                               ('input_len', [inlen_range]),
                           ]))

        position_ids = Tensor(name='position_ids',
                              dtype=trt.int32,
                              shape=[-1,2,-1],
                              dim_range=OrderedDict([
                                  ('batch_size', [bb_range]),
                                  ('2', [2]),
                                  ('input_len', [inlen_range]),
                              ]))

        for i in range(self._num_layers): 
            kv_dim_range = OrderedDict([
                ('batch_size', [bb_range]),
                ('kv', [2]),  #由于past_key_value 是一个tuple is k and v , each shape is :[b,]
                ('num_heads', [num_heads_kv]),# 
                ('past_key_len', [max_len_range]), # 解码与输入长度之和
                ('head_size', [head_size]),
            ])
            kv = Tensor(name=f'past_key_value_{i}',
                        dtype=self._kv_dtype,
                        shape=[-1, 2, num_heads_kv,-1,head_size],
                        dim_range=kv_dim_range)
            past_key_value.append(kv)

            # TODO(kaiyu): Remove this when TRT fix the named dimension
            assertion(shape(input_ids, 0) == shape(kv, 0), 'batch size')

        sequence_length = Tensor(
            name='sequence_length',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([('batch_size', [bb_range])]),
        )
        past_key_value_length = Tensor(
            name='past_key_value_length',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([('past_key_value_length', [max_len_range])]),
        )
        masked_tokens = Tensor(name='masked_tokens',
                               dtype=trt.int32,
                               shape=[-1, -1],
                               dim_range=OrderedDict([
                                   ('batch_size', [bb_range]),
                                   ('max_seq_len', [max_len_range]),
                               ]))

        input_lengths = Tensor(name='input_lengths',
                               dtype=trt.int32,
                               shape=[-1],
                               dim_range=OrderedDict([('batch_size', [bb_range])
                                                      ]))

        max_input_length = Tensor(name='max_input_length',
                                  dtype=trt.int32,
                                  shape=[-1],
                                  dim_range=OrderedDict([('max_input_len',
                                                          [inlen_range])]))

        last_token_ids = Tensor(name='last_token_ids',
                                dtype=trt.int32,
                                shape=[-1],
                                dim_range=OrderedDict([
                                    ('batch_size', [bb_range]),
                                ]))
        input_ids_ragged = RaggedTensor.from_row_lengths(
            input_ids, input_lengths, max_input_length) #input_length represent factual length

        cache_indirection = Tensor(name='cache_indirection',
                                   dtype=trt.int32,
                                   shape=[-1, -1, -1],
                                   dim_range=OrderedDict([
                                       ('batch_size', [bs_range]),
                                       ('beam_width', [beam_width_range]),
                                       ('max_seq_len', [max_len_range]),
                                   ]))

        return (input_ids_ragged, position_ids, past_key_value, sequence_length,
                past_key_value_length, masked_tokens, True, last_token_ids,
                cache_indirection)


from typing import List, Optional, Sequence, Tuple, Union
from functools import partial

class RmsNorm_my(Module):

    def __init__(self,
                 normalized_shape,
                 eps=1e-06,
                 elementwise_affine=True,
                 dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(shape=self.normalized_shape, dtype=dtype)
        else:
            self.register_parameter('weight', None)

        self.eps = eps
    def rms_norm_my(self,
            input: Tensor,
             normalized_shape: Union[int, Tuple[int]],
             weight: Optional[Tensor] = None,
             eps: float = 1e-06) -> Tensor:
        '''
        Add a RMS norm operation on a tensor.

        TODO: Document!
        '''
        normalized_shape = [normalized_shape] if isinstance(
            normalized_shape, int) else normalized_shape
        
        dim = tuple([-i - 1 for i in range(len(normalized_shape))])

        with precision("float32"):
            # input=cast(input, trt.float32)
            varx = pow(input, 2.0)
            varx = varx.mean(dim, keepdim=True)

            denom = varx + eps

            denom = denom.sqrt()

            y = input /denom

            if weight is not None:
                y = y * weight

        return y

    def forward(self, x):
        weight = None if self.weight is None else self.weight.value
        return self.rms_norm_my(x, self.normalized_shape, weight, self.eps)



