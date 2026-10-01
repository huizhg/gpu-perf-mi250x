# Each entry: (M, N, K, tag).  C[M, N] = A[M, K] @ B[K, N]
SHAPES = [
    # square: the classic benchmark
    *[(s, s, s, "square") for s in (512, 1024, 2048, 4096, 8192)],
    # prefill: many tokens (M = 2048) through 7B-model layers
    (2048, 4096, 4096, "prefill"),
    (2048, 11008, 4096, "prefill"),
    (2048, 4096, 11008, "prefill"),
    # decode: 1 to 16 tokens through the same layers
    (1, 4096, 4096, "decode"),
    (4, 4096, 4096, "decode"),
    (16, 4096, 4096, "decode"),
    (1, 11008, 4096, "decode"),
    # ESM-2 650M: one 1024-token protein through QKV, attn-out, FFN up, FFN down
    (1024, 3840, 1280, "esm"),
    (1024, 1280, 1280, "esm"),
    (1024, 5120, 1280, "esm"),
    (1024, 1280, 5120, "esm"),
    # odd sizes: not multiples of 128, to catch masking bugs later
    (1000, 1000, 1000, "odd"),
    (3000, 2000, 1500, "odd"),
]