import torch, triton, triton.language as tl

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