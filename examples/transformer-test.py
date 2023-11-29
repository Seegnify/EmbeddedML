import numpy as np
import math

import torch
#from torch.nn.functional import scaled_dot_product_attention

def my_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
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
    print("\nvalue:\n", value)
    return attn_weight @ value

def test_attention():
    # Set random seed for reproducibility
    np.random.seed(42)

    # input matrices
    Q = torch.tensor([
        [1.0,2,3],
        [4,5,6],
    ], requires_grad=True)
    K = torch.tensor([
        [0.1,0.2,0.3],
        [0.4,0.5,0.6],
        [1.4,1.5,1.6],
        [2.4,2.5,2.6],
    ], requires_grad=True)
    V = torch.tensor([
        [-2.0,7,8,2,2],
        [4,1,-9,3,3],
        [1,2,3,4,4],
        [4,5,6,5,5],
    ], requires_grad=True)
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

    attn_grad = torch.ones_like(attention);
    attn_grad[:,0] = 5
    print("attn_grad", attn_grad)
    attention.backward(attn_grad)

    print("\Q gradient:\n", Q.grad)
    print("\K gradient:\n", K.grad)
    print("\V gradient:\n", V.grad)

def test_softmax():
    print("=== test_softmax")
    # Q = torch.tensor([
    #     [1.0,2,3],
    #     [4,5,6],
    # ], requires_grad=True)
    Q = torch.tensor(
        [1.0,2,3],
        requires_grad=True)
    print("Q", Q)

    sQ = torch.softmax(Q, dim=-1)
    print("softmax(Q)", sQ)

    sQ_grad = torch.ones_like(sQ);
    sQ_grad[0] = 5
    print("softmax(Q).grad", sQ_grad)

    sQ.backward(sQ_grad)
    print("Q.grad", Q.grad)

def test_softmax_np():
    print("=== test_softmax_np")
    import numpy as np

    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # Subtracting np.max(x) for numerical stability
        return exp_x / np.sum(exp_x, axis=0)

    def softmax_derivative(x):
        s = softmax(x)
        return s * (np.eye(len(x)) - s.reshape(-1, 1))

    # Example usage:
    Q = np.array([1.0, 2.0, 3.0])
    print("Q", Q)
    soft = softmax(Q)
    soft_derivative = softmax_derivative(Q)

    print("Softmax:", soft)
    print("Softmax Derivative:\n", soft_derivative)

    soft_derivative = soft_derivative.transpose()
    soft_derivative = np.sum(soft_derivative, axis=-1)
    print("Softmax Derivative:\n", soft_derivative)

if __name__ == "__main__":
    test_attention()
    #test_softmax()
    #test_softmax_np()
