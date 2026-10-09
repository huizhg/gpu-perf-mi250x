MODELS = {"3b": (36, 2560), "650m": (33, 1280)}     # (layers, hidden size)

def flops_for_tokens(L, model="3b"):
    layers, d = MODELS[model]
    P = 12 * layers * d * d          # linear-layer weights: 4d² attention + 8d² FFN, per layer
    return 2 * P * L + 4 * layers * L * L * d

def flops_for_protein(n, model="3b"):
    return flops_for_tokens(min(n, 1022) + 2, model)