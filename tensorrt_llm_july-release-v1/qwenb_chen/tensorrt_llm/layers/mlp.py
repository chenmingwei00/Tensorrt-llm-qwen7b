from ..functional import ACT2FN
from ..module import Module
from .linear import ColumnLinear, RowLinear


class MLP(Module):

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 hidden_act,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1):
        super().__init__()
        if hidden_act not in ACT2FN:
            raise ValueError(
                'unsupported activation function: {}'.format(hidden_act))
        self.w1 = ColumnLinear(hidden_size,
                               ffn_hidden_size//2,
                               bias=None,
                               dtype=dtype,
                               tp_group=tp_group,
                               tp_size=tp_size,
                               gather_output=False)
        
        self.w2 = ColumnLinear(hidden_size,
                               ffn_hidden_size//2,
                               bias=None,
                               dtype=dtype,
                               tp_group=tp_group,
                               tp_size=tp_size,
                               gather_output=False)
        
        ff_dim_in = ffn_hidden_size // 2

        self.c_proj = RowLinear(ff_dim_in,
                              hidden_size,
                              bias=None,
                              dtype=dtype,
                              tp_group=tp_group,
                              tp_size=tp_size)
        self.hidden_act = hidden_act
        self.dtype = dtype

    def forward(self, hidden_states):
        a1 = self.w1(hidden_states)
        a2 = self.w2(hidden_states)
        intermediate_parallel = a1 * ACT2FN[self.hidden_act](a2)
        output = self.c_proj(intermediate_parallel)
        return output


class GatedMLP(MLP):

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 hidden_act,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1):
        super().__init__(hidden_size,
                         ffn_hidden_size,
                         hidden_act,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size)
        self.gate = ColumnLinear(hidden_size,
                                 ffn_hidden_size,
                                 bias=bias,
                                 dtype=dtype,
                                 tp_group=tp_group,
                                 tp_size=tp_size,
                                 gather_output=False)

    def forward(self, hidden_states):
        inter = self.fc(hidden_states)
        inter = ACT2FN[self.hidden_act](inter)
        gate = self.gate(hidden_states)
        output = self.proj(inter * gate)
        return output
