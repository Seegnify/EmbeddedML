import numpy as np
import math

import torch
#from torch.nn.functional import scaled_dot_product_attention
import transformer

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

def test_attention():
    print("=== test_attention")
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
    attn_grad[0,0] = 5
    print("attn_grad", attn_grad)
    attention.backward(attn_grad)

    print("\Q gradient:\n", Q.grad)
    print("\K gradient:\n", K.grad)
    print("\V gradient:\n", V.grad)

def test_multihead_attention():
    print("=== test_multihead_attention")
    # Define some parameters
    batch_size = 1
    sequence_length = 3
    embed_size = 4
    num_heads = 2

    # Create input tensors
    query = torch.rand(batch_size, sequence_length, embed_size)
    key = torch.rand(batch_size, sequence_length, embed_size)
    value = torch.rand(batch_size, sequence_length, embed_size)

    # Load inputs
    input_path = "attention_input.pt"
    try:
        inputs = torch.load(input_path)
        query = inputs["query"]
        key = inputs["key"]
        value = inputs["value"]
        print("Inputs loaded from", input_path)
    except:
        inputs = {}
        inputs["query"] = query
        inputs["key"] = key
        inputs["value"] = value
        torch.save(inputs, input_path)
        print("New inputs saved to", input_path)

    # Create MultiheadAttention module
    attention = torch.nn.MultiheadAttention(embed_size, num_heads, bias=True)

    # Load model
    model_path = "attention_model.pt"
    try:
        attention.load_state_dict(torch.load(model_path))
        print("Model loaded from", model_path)
    except:
        torch.save(attention.state_dict(), model_path)
        print("New model saved to", model_path)

    """
    bias = attention.state_dict()["in_proj_bias"]
    bias[0:bias.shape[0]] = torch.tensor([
        0.4755, 0.1042, 0.6459, 0.2230,      
        0.0739, 0.6705, 0.8532, 0.7830,
        0.1097, 0.8451, 0.7208, 0.2440
    ])

    bias = attention.state_dict()["out_proj.bias"]
    bias[0:bias.shape[0]] = torch.tensor([
        0.0307, 0.1667, 0.4442, 0.1971
    ])
    """

    print("Start Attention Params")
    for name in attention.state_dict():
        print("name:", name)
        param = attention.state_dict()[name]
        print("shape:", param.shape)
        print(param)
        if param.detach().numpy().sum() == 0:
          param[0:param.shape[0]] = torch.rand(1, param.shape[0]) 
    print("End Attention Params")

    # Apply Multihead Attention
    output, attention_weights = attention(query, key, value)

    # Print shapes of input and output tensors
    print("Query:", query.shape)
    print(query)
    print("Key:", key.shape)
    print(key)
    print("Value:", value.shape)
    print(value)
    print("Output:", output.shape)
    print(output)
    print("Attention weights:", attention_weights.shape)    
    print(attention_weights)

def test_my_multihead_attention():
    print("=== test_my_multihead_attention")
    embed_size = 4
    num_heads = 2
        
    attention = transformer.MultiHeadAttention(embed_size, num_heads)

    q = torch.tensor(
        [[[0.0878, 0.0416, 0.6166, 0.1477],
         [0.9752, 0.8866, 0.5407, 0.1911],
         [0.5300, 0.2800, 0.5306, 0.4950]]],
         requires_grad=True
    )
    k = torch.tensor(
        [[[0.2248, 0.4832, 0.5916, 0.0345],
         [0.4916, 0.0881, 0.3768, 0.3048],
         [0.0780, 0.3594, 0.0297, 0.6474]]],
         requires_grad=True
    )
    v = torch.tensor(
        [[[0.2014, 0.0033, 0.2326, 0.5677],
         [0.6842, 0.1161, 0.8033, 0.6450],
         [0.4097, 0.3034, 0.8000, 0.7103]]],
         requires_grad=True
    ) 

    params = attention.state_dict()
    params["W_q.weight"][:,:] = torch.tensor(
        [[ 0.4271,  0.3013, -0.4279, -0.2122],
        [ 0.2983,  0.3350, -0.4619,  0.5432],
        [-0.1488,  0.1778, -0.4288, -0.5003],
        [ 0.1173,  0.3713, -0.2347, -0.2251]]    
    )
    params["W_k.weight"][:,:] = torch.tensor(
        [[ 0.1557,  0.4673,  0.0920,  0.3889],
        [ 0.5867,  0.0088,  0.4371,  0.0371],
        [ 0.4897, -0.0109, -0.0646,  0.5190],
        [-0.5768,  0.1376, -0.5507,  0.5315]],
    )
    params["W_v.weight"][:,:] = torch.tensor(
        [[-0.3599, -0.4841,  0.0526, -0.5235],
        [-0.1576,  0.4844, -0.3817,  0.2549],
        [-0.1432,  0.5141, -0.5741, -0.0179],
        [-0.0103, -0.4235, -0.5195, -0.1589]]
    )
    params["W_o.weight"][:,:] = torch.tensor(
        [[-0.2588,  0.4873,  0.0642,  0.4206],
        [ 0.3272,  0.3202,  0.4458, -0.3825],
        [-0.4631, -0.2740, -0.2628, -0.4749],
        [-0.3654,  0.4841,  0.4618, -0.1188]]
    )

    params["W_q.bias"][:] = torch.tensor(
        [0.4755, 0.1042, 0.6459, 0.2230]
    )
    params["W_k.bias"][:] = torch.tensor(
        [0.0739, 0.6705, 0.8532, 0.7830]
    )
    params["W_v.bias"][:] = torch.tensor(
        [0.1097, 0.8451, 0.7208, 0.2440]
    )
    params["W_o.bias"][:] = torch.tensor(
        [0.0307, 0.1667, 0.4442, 0.1971]
    )

    print("Start Attention Params")
    for name in attention.state_dict():
        print("name:", name)
        param = attention.state_dict()[name]
        print("shape:", param.shape)
        print(param)
        if param.detach().numpy().sum() == 0:
          param[0:param.shape[0]] = torch.rand(1, param.shape[0]) 
    print("End Attention Params")
    
    A = attention(q, k, v)
    print("Attention output")
    print(A)

    dA = torch.ones_like(A)
    dA[0,0,0] = 5
    print("dA", dA)

    A.backward(dA)
    print("dAdq", q.grad)
    print("dAdk", k.grad)
    print("dAdv", v.grad)

    for p in params:
      print(p, "grad", params[p].grad)


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

def test_normlayer():
    print("=== test_normlayer")
    Q = torch.tensor([
        [1.0,2,3],
        [4,5,6],
    ], requires_grad=True)
    print("Q", Q)

    N = torch.nn.LayerNorm(Q.shape)
    NQ = N(Q)
    print("N", NQ)

    NQ_grad = torch.zeros_like(NQ);
    NQ_grad[0][0] = 1
    print("N(Q).grad", NQ_grad)

    NQ.backward(NQ_grad)
    print("dNdQ.grad", Q.grad)

if __name__ == "__main__":
    #test_softmax()
    test_attention()
    #test_multihead_attention()
    test_my_multihead_attention()
    #test_normlayer()
