"""
Fused attention from triton tutorial.
Modified from the original implementation
 - https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py
===============

This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)
"""

import torch
import triton
import triton.language as tl


# yapf: disable
@triton.jit
def fused_attention_kernel(
        Out, L, M,  # outputs
        Q, K, V,
        sm_scale,
        seq_len,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)  # External circulation represent Tr
    off_hz = tl.program_id(1)  # Internal circulation   which ids ,all ids is Tc
    stride_h = BLOCK_DMODEL * seq_len  # each block seq_len not entire seq_len

    # initialize offsets BLOCK_M is likely to represent flash attention's Br  BLOCK_N represent attention's Bc
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)  # the d
    off_q = off_hz * stride_h + offs_m[:, None] * BLOCK_DMODEL + offs_d[None,
                                                                 :]  # off_hz * stride_h  which number does it represent
    off_k = off_hz * stride_h + offs_n[None, :] * BLOCK_DMODEL + offs_d[:, None]
    off_v = off_hz * stride_h + offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :]
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q  # current ptrs
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # Br
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)  # shape [Br,d]
    # loop over k, v and update accumulator
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        # -- compute qk ----
        k = tl.load(k_ptrs)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        # compute new m
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)  # Br
        # correct old l
        l_prev *= tl.exp(m_prev - m_curr)
        # attention weights
        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        # rescale operands of matmuls
        l_rcp = 1. / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        # update acc
        p = p.to(Q.dtype.element_ty)
        v = tl.load(v_ptrs)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr
        # update pointers
        k_ptrs += BLOCK_N * BLOCK_DMODEL
        v_ptrs += BLOCK_N * BLOCK_DMODEL
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    l_ptrs = L + off_hz * seq_len + offs_m
    m_ptrs = M + off_hz * seq_len + offs_m
    tl.store(l_ptrs, l_prev)
    tl.store(m_ptrs, m_prev)
    # initialize pointers to output
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = off_hz * stride_h + offs_m[:, None] * BLOCK_DMODEL + offs_n[None, :]
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)


def fused_attention(q, k, v, sm_scale, o_buf=None, l_buf=None, m_buf=None):
    BLOCK = 16 if q.dtype == torch.float16 else 64
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q) if o_buf is None else o_buf
    grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
    shape = (q.shape[0] * q.shape[1], q.shape[2])
    L = torch.empty(shape, device=q.device, dtype=torch.float32) if l_buf is None else l_buf
    m = torch.empty(shape, device=q.device, dtype=torch.float32) if m_buf is None else m_buf
    num_warps = 4 if Lk <= 64 else 8

    fused_attention_kernel[grid](
        o, L, m,
        q, k, v,
        sm_scale, q.shape[2],
        # tl.constexpr
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        BLOCK_DMODEL=Lk,
        num_warps=num_warps
    )

    return o


# yapf: enable

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, o_buf=None, l_buf=None, m_buf=None):
        BLOCK = 32 if q.dtype == torch.float16 else 64
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q) if o_buf is None else o_buf
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
        shape = (q.shape[0] * q.shape[1], q.shape[2])
        L = torch.empty(shape, device=q.device, dtype=torch.float32) if l_buf is None else l_buf
        m = torch.empty(shape, device=q.device, dtype=torch.float32) if m_buf is None else m_buf
        num_warps = 4 if Lk <= 64 else 8

        fused_attention_kernel[grid](
            o, L, m,
            q, k, v,
            sm_scale, q.shape[2],
            # tl.constexpr
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            BLOCK_DMODEL=Lk,
            num_warps=num_warps
        )

        return o


attention = _attention.apply

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func

    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

BATCH, N_HEADS, N_CTX, D_HEAD = 1, 32, 512, 128
# vary seq length for fixed head and batch=4
configs = [
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[2 ** i for i in range(10, 15)],
        line_arg="provider",
        line_vals=["triton"] + (["flash"] if HAS_FLASH else []),
        line_names=["Triton"] + (["Flash-2"] if HAS_FLASH else []),
        styles=[("red", "-"), ("blue", "-")],
        ylabel="ms",
        plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}",
        args={
            "H": N_HEADS,
            "BATCH": BATCH,
            "D_HEAD": D_HEAD,
            "dtype": torch.float16,
            "mode": mode,
            "causal": causal,
        },
    )
    for mode in ["fwd"]
    for causal in [True]
]


@triton.testing.perf_report(configs)
def bench_flash_attention(
        BATCH, H, N_CTX, D_HEAD, causal, mode, provider, dtype=torch.float16, device="cuda"
):
    assert mode in ["fwd"]
    warmup = 25
    rep = 100
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn(
            (BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True
        )
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9

# bench_flash_attention.run(save_path=".", print_data=True)
