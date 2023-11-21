import numpy as np
import math

import torch
#from torch.nn.functional import scaled_dot_product_attention

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    print("L=", L, "S=", S, "D=", query.size(-1))
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            print("\nattn_mask logical_not:\n", attn_mask.logical_not())
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            print("\nattn_mask filled with -Inf:\n", attn_mask)
        else:
            print("\nattn_mask unchanged:\n", attn_mask)
            attn_bias += attn_mask
    print("\nattn_bias:\n", attn_bias)
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    print("\nattention (A before softmax):\n", attn_weight)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    print("\nattention (A after softmax):\n", attn_weight)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # input matrices
    Q = torch.Tensor([
        [1,2,3],
        [4,5,6],
    ])
    K = torch.Tensor([
        [0.1,0.2,0.3],
        [0.4,0.5,0.6],
        [1.4,1.5,1.6],
        [2.4,2.5,2.6],
    ])
    V = torch.Tensor([
        [-2,7,8,2,2],
        [4,1,-9,3,3],
        [1,2,3,4,4],
        [4,5,6,5,5],
    ])
    M = torch.tensor([
        [1,1,1,1],
        [1,1,0,0],
    ], dtype=torch.bool)

    dropout = 0.0

    # Display the input and output matrices
    print("\nQuery Matrix (Q):\n", Q, Q.dtype)
    print("\nKey Matrix (K):\n", K, K.dtype)
    print("\nValue Matrix (V):\n", V, V.dtype)
    #print("\nMask Matrix (V):\n", M, M.dtype)

    # Apply multihead attention
    attention = scaled_dot_product_attention(Q, K, V, M, dropout)

    print("\nSinglehead Attention Output:\n", attention)

if __name__ == "__main__":
    main()
