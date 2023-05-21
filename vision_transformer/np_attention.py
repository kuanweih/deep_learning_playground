import numpy as np


if __name__ == "__main__":

    image_size = 224
    patch_size = 16
    num_layers = 2  # default 12
    num_heads = 12
    hidden_dim = 768
    mlp_dim = 3072

    np.random.seed(1224)

    # fake input data at the input embedding level.
    batch_size = 4
    seq_length = 197  # 14 * 14 + 1 = 197
    x = np.random.uniform(low=-0.5, high=0.5, size=(batch_size, seq_length, hidden_dim))  # (batch, seq, d_model) = (4, 197, 768)

    # attention layer
    attn_dim = 64
    # head 0
    W_Q = np.random.uniform(low=-0.5, high=0.5, size=(hidden_dim, attn_dim))  # (768, 64)
    W_K = np.random.uniform(low=-0.5, high=0.5, size=(hidden_dim, attn_dim))  # (768, 64)
    W_V = np.random.uniform(low=-0.5, high=0.5, size=(hidden_dim, attn_dim))  # (768, 64)
    Q = x @ W_Q  # (4, 197, 64)
    K = x @ W_K  # (4, 197, 64)
    V = x @ W_V  # (4, 197, 64)
    score = Q @ np.transpose(K, axes=[0, 2, 1])  # (4, 197, 197)
    score = score / np.sqrt(attn_dim)  # (4, 197, 197)
    score = np.exp(score - np.max(score)) / np.exp(score - np.max(score)).sum(axis=2, keepdims=True)  # softmax: (4, 197, 197)
    z = score @ V   # (4, 197, 64)
    # multi-head attn: 12 heads. let's just mock them! ps: 12 * 64 = 768
    z = np.concatenate([z for _ in range(num_heads)], axis=2)   # (4, 197, 768)
    # residual
    x = x + z  # (4, 197, 768)

    # MLP layer
    # layer norm
    z = (x - x.mean(axis=2, keepdims=True)) / np.sqrt(x.var(axis=2, keepdims=True) + 1e-6)  # (4, 197, 768)
    # mlp 1
    M_1 = np.random.uniform(low=-0.5, high=0.5, size=(hidden_dim, mlp_dim))  # (768, 3072)
    z = z @ M_1  # (4, 197, 3072)
    # relu
    z = np.maximum(0, z)  # (4, 197, 3072)
    # mlp 2
    M_2 = np.random.uniform(low=-0.5, high=0.5, size=(mlp_dim, hidden_dim))  # (3072, 768)
    z = z @ M_2  # (4, 197, 768)
    # residual
    x = x + z  # (4, 197, 768)
