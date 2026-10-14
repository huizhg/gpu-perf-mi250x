"""v5: your v4 GEMM with an optional fused epilogue, y = GELU(x @ W^T + b), exact (erf) GELU.

One kernel, two compiled versions, switched by the constexpr FUSE_GELU:
  gemm_v5_gelu(a, b, bias)  -> GEMM + bias + exact GELU in the epilogue (the fused contestant)
  gemm_v5_plain(a, b)       -> the same GEMM with the epilogue compiled out ("your GEMM alone")
Because both come from the same source, any time difference between them is the epilogue's cost.
"""
import itertools, torch, triton, triton.language as tl

try:
    from kernels.gemm import CONFIGS_V4 as CONFIGS        # the same search space as your v4
except ImportError:
    TILES = [(64, 64), (128, 128), (128, 256), (256, 128)]
    CONFIGS = [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8,
                       "matrix_instr_nonkdim": 16},
                      num_warps=nw, num_stages=2)
        for (bm, bn), bk, nw in itertools.product(TILES, [32, 64], [4, 8])
    ]


@triton.jit
def gemm_kernel_v5(a_ptr, b_ptr, c_ptr, bias_ptr, M, N, K,
                   s_am, s_ak, s_bk, s_bn, s_cm, s_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   GROUP_M: tl.constexpr, FUSE_GELU: tl.constexpr):
    # v2 grouped ordering: neighbouring programs share rows of A in L2
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_in_group = GROUP_M * num_pid_n
    first_m = (pid // num_in_group) * GROUP_M
    group_size_m = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % num_in_group) % group_size_m
    pid_n = (pid % num_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * s_am + offs_k[None, :] * s_ak
    b_ptrs = b_ptr + offs_k[:, None] * s_bk + offs_n[None, :] * s_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k0 + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k0 + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * s_ak
        b_ptrs += BLOCK_K * s_bk

    # Epilogue: runs on the fp32 accumulator while it is still in registers.
    # FUSE_GELU is a compile-time constant, so with False these lines don't exist in the binary.
    if FUSE_GELU:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]
        acc = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865476))   # exact GELU; if missing: tl.math.erf

    c_ptrs = c_ptr + offs_m[:, None] * s_cm + offs_n[None, :] * s_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)   # bf16 in -> bf16 out


# Two autotuners over the same kernel, so the fused and plain versions each pick their own best config.
_gemm_gelu = triton.autotune(configs=CONFIGS, key=["M", "N", "K"])(gemm_kernel_v5)
_gemm_plain = triton.autotune(configs=CONFIGS, key=["M", "N", "K"])(gemm_kernel_v5)


def _launch(kernel, a, b, bias, fuse):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"inner dims differ: {K} vs {K2}"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    kernel[grid](a, b, c, bias if bias is not None else c, M, N, K,
                 a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                 FUSE_GELU=fuse)
    return c


def gemm_v5_gelu(a, b, bias):
    """GELU(a @ b + bias). b may be a transposed view such as w.t(); the strides handle it."""
    return _launch(_gemm_gelu, a, b, bias, True)


def gemm_v5_plain(a, b):
    """a @ b with the epilogue compiled out: your GEMM alone."""
    return _launch(_gemm_plain, a, b, None, False)