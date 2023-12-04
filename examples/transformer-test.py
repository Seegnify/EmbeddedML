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
    attention = torch.nn.MultiheadAttention(embed_size, num_heads)

    # Load model
    model_path = "attention_model.pt"
    try:
        attention.load_state_dict(torch.load(model_path))
        print("Model loaded from", model_path)
    except:
        torch.save(attention.state_dict(), model_path)
        print("New model saved to", model_path)

    print("Start Attention Params")
    for name, param in attention.named_parameters():
        print(f"Parameter name: {name}, Shape: {param.shape}")
        print(param)
        #np_param = param.detach().numpy()
        #print(np_param)
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

if __name__ == "__main__":
    #test_softmax()
    #test_attention()
    test_multihead_attention()
