import torch, triton, triton.language as tl
import itertools


@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    s_am, s_ak, s_bk, s_bn, s_cm, s_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0) # row tile index
    pid_n = tl.program_id(1) # col tile index
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) # my row of C 
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) # my col of C
    offs_k = tl.arange(0, BLOCK_K) # position within one K chunk

    # pointer grids for the first K chunk of A and B
    a_ptrs = a_ptr + offs_m[:, None] * s_am + offs_k[None, : ] * s_ak # shape [BLOCK_M, BLOCK_K]
    b_ptrs = b_ptr + offs_k[:, None] * s_bk + offs_n[None, : ] * s_bn # shape [BLOCK_K, BLOCK_N]

    # FP32 accumultor: summing thousands of FP16 products is FP16 would lose precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:,None] < M) & (k0 + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None,:] < N) & (k0 + offs_k[:, None] < K), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * s_ak
        b_ptrs += BLOCK_K * s_bk
    c_ptrs = c_ptr + offs_m[:, None] * s_cm + offs_n[None, : ] * s_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:,None] < M) & (offs_n[None, :] < N))

def gemm(a, b, BM=128, BN=128, BK=32):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "inner dimensions must match"
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN)) # one program per output tile
    gemm_kernel[grid](a, b, c, M, N, K,
                      a.stride(0), a.stride(1),
                      b.stride(0), b.stride(1),
                      c.stride(0), c.stride(1),
                      BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    return c

@triton.jit
def gemm_kernel_v2(
    a_ptr, b_ptr, c_ptr, M, N, K,
    s_am, s_ak, s_bk, s_bn, s_cm, s_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_in_group = GROUP_M * num_pid_n # programs per group of group_m tile-rows
    group_id = pid // num_in_group 
    first_m = group_id * GROUP_M 
    group_size_m = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % num_in_group) % group_size_m
    pid_n = (pid % num_in_group) // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) # my row of C 
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) # my col of C
    offs_k = tl.arange(0, BLOCK_K) # position within one K chunk

    # pointer gri sds for the first K chunk of A and B
    a_ptrs = a_ptr + offs_m[:, None] * s_am + offs_k[None, : ] * s_ak # shape [BLOCK_M, BLOCK_K]
    b_ptrs = b_ptr + offs_k[:, None] * s_bk + offs_n[None, : ] * s_bn # shape [BLOCK_K, BLOCK_N]

    # FP32 accumultor: summing thousands of FP16 products is FP16 would lose precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:,None] < M) & (k0 + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None,:] < N) & (k0 + offs_k[:, None] < K), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * s_ak
        b_ptrs += BLOCK_K * s_bk
    c_ptrs = c_ptr + offs_m[:, None] * s_cm + offs_n[None, : ] * s_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:,None] < M) & (offs_n[None, :] < N))
   
def gemm_v2(a, b, BM=128, BN=128, BK=32, GM=8, NW=4, NS=2, **opts):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "inner dimensions must match"
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),) 
    gemm_kernel_v2[grid](a, b, c, M, N, K,
                         a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                         BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
                         num_warps=NW, num_stages=NS, **opts)
    return c

# Autotuning

TILES = [(32, 64), (64, 64), (64, 128), (128, 128), (128, 256), (256, 128)]
CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
                  num_warps=nw, num_stages=2)
    for (bm, bn), bk, nw in itertools.product(TILES, [32, 64], [4, 8])
]   # 6 tiles x 2 BLOCK_K x 2 warp counts = 24 candidates

# Re-tune whenever M, N or K changes
gemm_kernel_v3 = triton.autotune(configs=CONFIGS, key=["M", "N", "K"])(gemm_kernel_v2)

def gemm_v3(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # The grid depends on the tile size, which isn't known until a config is chosen
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    gemm_kernel_v3[grid](a, b, c, M, N, K,
                         a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c

@triton.jit
def gemm_kernel_v1_rows(
    a_ptr, b_ptr, c_ptr, M, N, K,
    s_am, s_ak, s_bk, s_bn, s_cm, s_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(1) # row tile index
    pid_n = tl.program_id(0) # col tile index
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) # my row of C 
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) # my col of C
    offs_k = tl.arange(0, BLOCK_K) # position within one K chunk

    # pointer grids for the first K chunk of A and B
    a_ptrs = a_ptr + offs_m[:, None] * s_am + offs_k[None, : ] * s_ak # shape [BLOCK_M, BLOCK_K]
    b_ptrs = b_ptr + offs_k[:, None] * s_bk + offs_n[None, : ] * s_bn # shape [BLOCK_K, BLOCK_N]

    # FP32 accumultor: summing thousands of FP16 products is FP16 would lose precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:,None] < M) & (k0 + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None,:] < N) & (k0 + offs_k[:, None] < K), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * s_ak
        b_ptrs += BLOCK_K * s_bk
    c_ptrs = c_ptr + offs_m[:, None] * s_cm + offs_n[None, : ] * s_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:,None] < M) & (offs_n[None, :] < N))

def gemm_v1_rows(a, b, BM=128, BN=128, BK=32):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "inner dimensions must match"
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(N, BN), triton.cdiv(M, BM)) # one program per output tile
    gemm_kernel_v1_rows[grid](a, b, c, M, N, K,
                      a.stride(0), a.stride(1),
                      b.stride(0), b.stride(1),
                      c.stride(0), c.stride(1),
                      BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    return c

CONFIGS_V4 = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8,
                   "matrix_instr_nonkdim": 16},
                  num_warps=nw, num_stages=2)
    for (bm, bn), bk, nw in itertools.product(TILES, [32, 64], [4, 8])
]
gemm_kernel_v4 = triton.autotune(configs=CONFIGS_V4, key=["M", "N", "K"])(gemm_kernel_v2)

def gemm_v4(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    gemm_kernel_v4[grid](a, b, c, M, N, K,
                         a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c