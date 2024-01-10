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


def test_scaled_dot_product_attention():
    print("=== test_scaled_dot_product_attention")
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
    attn = scaled_dot_product_attention(Q, K, V, M, dropout)

    print("Output:\n", attn)

    attn_grad = torch.ones_like(attn);
    attn_grad[0,0] = 5
    print("attn_grad", attn_grad)
    attn.backward(attn_grad)

    print("Q gradient:\n", Q.grad)
    print("K gradient:\n", K.grad)
    print("V gradient:\n", V.grad)


def test_multihead_attention():
    print("=== test_multihead_attention")
    # Define some parameters
    batch_size = 1
    sequence_length = 3
    embed_size = 4
    num_heads = 2
    dropout = 0.0

    # Create input tensors
    q = torch.rand(batch_size, sequence_length, embed_size)
    k = torch.rand(batch_size, sequence_length, embed_size)
    v = torch.rand(batch_size, sequence_length, embed_size)

    # Load inputs
    """
    input_path = "attention_input.pt"
    try:
        inputs = torch.load(input_path)
        q = inputs["query"]
        k = inputs["key"]
        v = inputs["value"]
        print("Inputs loaded from", input_path)
    except:
        inputs = {}
        inputs["query"] = q
        inputs["key"] = k
        inputs["value"] = v
        torch.save(inputs, input_path)
        print("New inputs saved to", input_path)
    """

    q = torch.tensor(
        [[[110.0878, 0.0416, 0.6166, 0.1477],
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
    
    print("q", q) 
    print("k", k) 
    print("v", v) 

    # Create MultiheadAttention module
    attention = torch.nn.MultiheadAttention(embed_size, num_heads, 
    bias=True, dropout=dropout)

    # Load model
    """
    model_path = "attention_model.pt"
    try:
        attention.load_state_dict(torch.load(model_path))
        print("Model loaded from", model_path)
    except:
        torch.save(attention.state_dict(), model_path)
        print("New model saved to", model_path)
    """

    weight = attention.state_dict()["in_proj_weight"]
    weight[:,:] = torch.tensor([
        [ 0.4271,  0.3013, -0.4279, -0.2122],
        [ 0.2983,  0.3350, -0.4619,  0.5432],
        [-0.1488,  0.1778, -0.4288, -0.5003],
        [ 0.1173,  0.3713, -0.2347, -0.2251],

        [ 0.1557,  0.4673,  0.0920,  0.3889],
        [ 0.5867,  0.0088,  0.4371,  0.0371],
        [ 0.4897, -0.0109, -0.0646,  0.5190],
        [-0.5768,  0.1376, -0.5507,  0.5315],

        [-0.3599, -0.4841,  0.0526, -0.5235],
        [-0.1576,  0.4844, -0.3817,  0.2549],
        [-0.1432,  0.5141, -0.5741, -0.0179],
        [-0.0103, -0.4235, -0.5195, -0.1589]
    ])

    bias = attention.state_dict()["in_proj_bias"]
    bias[0:bias.shape[0]] = torch.tensor([
        0.4755, 0.1042, 0.6459, 0.2230,      
        0.0739, 0.6705, 0.8532, 0.7830,
        0.1097, 0.8451, 0.7208, 0.2440
    ])

    weight = attention.state_dict()["out_proj.weight"]
    weight[:,:] = torch.tensor([
        [-0.2588,  0.4873,  0.0642,  0.4206],
        [ 0.3272,  0.3202,  0.4458, -0.3825],
        [-0.4631, -0.2740, -0.2628, -0.4749],
        [-0.3654,  0.4841,  0.4618, -0.1188]
    ])

    bias = attention.state_dict()["out_proj.bias"]
    bias[0:bias.shape[0]] = torch.tensor([
        0.0307, 0.1667, 0.4442, 0.1971
    ])

    print("Start Attention Params")
    for name in attention.state_dict():
        print("name:", name)
        param = attention.state_dict()[name]
        print("shape:", param.shape)
        print(param)
    print("End Attention Params")

    # Apply Multihead Attention
    A, attention_weights = attention(q, k, v,
      need_weights=False, average_attn_weights=False)

    # Print shapes of input and output tensors
    print("Output:", A.shape)
    print(A)
    print("Attention weights:")    
    print(attention_weights)

    dA = torch.ones_like(A)
    dA[0,0,0] = 5
    print("dA", dA)

    A.backward(dA)
    print("dAdq", q.grad)
    print("dAdk", k.grad)
    print("dAdv", v.grad)

    # Collect gradients
    gradients = {}
    for name, param in attention.named_parameters():
        gradients[name] = param.grad.clone()

    # Print or use the gradients as needed
    for name, grad in gradients.items():
        print(f'Gradient for {name}:\n{grad}')


def test_my_multihead_attention():
    print("=== test_my_multihead_attention")
    embed_size = 4
    num_heads = 2
    dropout = 0.0
        
    attention = transformer.MultiHeadAttention(embed_size, num_heads, dropout)

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
    print("q", q) 
    print("k", k) 
    print("v", v) 

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

    print("Start MultiHeadAttention Params")
    for name in params:
        print("name:", name)
        param = params[name]
        print("shape:", param.shape)
        print(param, "requires_grad", param.requires_grad)
        if param.detach().numpy().sum() == 0:
          param[0:param.shape[0]] = torch.rand(1, param.shape[0]) 
    print("End MultiHeadAttention Params")
    
    A = attention(q, k, v)
    print("MultiHeadAttention output")
    print(A)

    dA = torch.ones_like(A)
    dA[0,0,0] = 5
    print("dA", dA)

    A.backward(dA)
    print("dAdq", q.grad)
    print("dAdk", k.grad)
    print("dAdv", v.grad)

    # Collect gradients
    gradients = {}
    for name, param in attention.named_parameters():
        gradients[name] = param.grad.clone()

    # Print or use the gradients as needed
    for name, grad in gradients.items():
        print(f'Gradient for {name}:\n{grad}')


def test_position_wise_feed_forward():
    print("=== test_position_wise_feed_forward")
    embed_size = 4
    hidden_size = 3
    dropout = 0.0

    model = transformer.PositionWiseFeedForward(embed_size, hidden_size, dropout)

    x = torch.tensor(
        [[[0.0878, 0.0416, 0.6166, 0.1477],
         [0.5300, 0.2800, 0.5306, 0.4950]]],
         requires_grad=True
    )

    params = model.state_dict()
    params["w_1.weight"][:,:] = torch.tensor(
        [[-0.3883,  0.2742, -0.4652, -0.1417],
        [-0.0996, -0.4170, -0.0302,  0.1254],
        [-0.2065,  0.0107,  0.3998,  0.3775]]
    )
    params["w_2.weight"][:,:] = torch.tensor(
        [[ 0.0348,  0.3779, -0.5751],
        [-0.0708, -0.4522, -0.4000],
        [ 0.3196,  0.2163,  0.5397],
        [-0.1805,  0.0472, -0.4630]],
    )
    params["w_1.bias"][:] = torch.tensor(
        [ 0.4282,  0.2099, -0.2209]
    )
    params["w_2.bias"][:] = torch.tensor(
        [-0.4660, -0.4707,  0.4046, -0.4392]
    )

    print("Start PositionwiseFeedForward Params")
    for name in params:
        print("name:", name)
        param = params[name]
        print("shape:", param.shape)
        print(param)
        if param.detach().numpy().sum() == 0:
          param[0:param.shape[0]] = torch.rand(1, param.shape[0]) 
    print("End PositionwiseFeedForward Params")

    A = model(x)
    print("PositionwiseFeedForward output")
    print(A)

    dA = torch.ones_like(A)
    dA[0,0,0] = 5
    print("dA", dA)

    A.backward(dA)
    print("dAdx", x.grad)

    # Collect gradients
    gradients = {}
    for name, param in model.named_parameters():
        gradients[name] = param.grad.clone()

    # Print or use the gradients as needed
    for name, grad in gradients.items():
        print(f'Gradient for {name}:\n{grad}')

def test_positional_encoding():
    print("=== test_positional_encoding")
    emb_size = 4
    seq_size = 5
    max_seq_size = 7

    model = transformer.PositionalEncoding(emb_size, max_seq_size)
    params = model.state_dict()

    x = torch.tensor(
        [[[1,2,3,4],
          [5,6,7,8],
          [0.0878, 0.0416, 0.6166, 0.1477],
          [-0.3883,  0.2742, -0.4652, -0.1417],
          [0.5300, 0.2800, 0.5306, 0.4950]]],
         requires_grad=True
    )
    print("x", x)

    print("Start Params")
    for name in params:
        print("name:", name)
        param = params[name]
        print("shape:", param.shape)
        print(param)
        if param.detach().numpy().sum() == 0:
          param[0:param.shape[0]] = torch.rand(1, param.shape[0])
    print("End Params")

    A = model(x)
    print("output")
    print(A)

    dA = torch.ones_like(A)
    dA[0,0,0] = 5
    print("dA", dA)

    A.backward(dA)
    print("dAdx", x.grad)

    # Collect gradients
    gradients = {}
    for name, param in model.named_parameters():
        gradients[name] = param.grad.clone()

    # Print or use the gradients as needed
    for name, grad in gradients.items():
        print(f'Gradient for {name}:\n{grad}')


def test_grad():
  import torch
  import torch.nn as nn

  # Define a simple model
  class SimpleModel(nn.Module):
    def __init__(self):
      super(SimpleModel, self).__init__()
      self.linear = nn.Linear(2, 2)

    def forward(self, x):
      return self.linear(x)

  model = SimpleModel()

  # Input data
  input_data = torch.randn(1, 2, requires_grad=True)

  # Forward pass
  output = model(input_data)
  print(f'SimpleModel output')
  print(output)

  print("Start SimpleModel Params")
  params = model.state_dict()
  for name in params:
      print("name:", name)
      param = params[name]
      print("shape:", param.shape)
      print(param)
      if param.detach().numpy().sum() == 0:
        param[0:param.shape[0]] = torch.rand(1, param.shape[0]) 
  print("End SimpleModel Params")

  # Create a dummy loss (you can use your actual loss function)
  loss = torch.sum(output)
  print(f'SimpleModel loss')
  print(loss)

  dL = torch.ones_like(loss)
  print("dL.shape", dL.shape)

  # Backward pass to compute gradients
  loss.backward(dL)

  # Collect gradients
  gradients = {}
  for name, param in model.named_parameters():
      gradients[name] = param.grad.clone()

  # Print or use the gradients as needed
  for name, grad in gradients.items():
      print(f'Gradient for {name}:\n{grad}')

  for p in params:
    print(p, "Missing Grad", params[p].requires_grad, params[p].grad)

  # Clear gradients for the next iteration (optional)
  model.zero_grad()


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

def test_layernorm():
    print("=== test_layernorm")
    x = torch.tensor(
         [[[109.6705, -10.2022,   8.5515,  11.0067],
         [113.4968,  -6.2074,  12.7281,  14.8871],
         [109.5167, -12.0805,   5.4300,   7.6243],
         [109.1908, -11.8192,   4.1436,   7.4589],
         [109.9948, -11.8377,   5.3094,   7.9933]]]    
         , requires_grad=True)
    print("x", x.shape, x)

    model = torch.nn.LayerNorm(x.shape[-1]) # row-wise norm

    print("Start Params")
    params = model.state_dict()    
    for name in params:
        print("name:", name)
        param = params[name]
        print("shape:", param.shape)
        print(param)
    print("End Params")

    A = model(x)
    print("output")
    print(A)

    dA = torch.ones_like(A)
    dA[0,0] = 5
    print("dA", dA)

    A.backward(dA)
    print("dAdx", x.grad)
    print("x.grad", x.grad)

    # Collect gradients
    gradients = {}
    for name, param in model.named_parameters():
        gradients[name] = param.grad.clone()

    # Print or use the gradients as needed
    for name, grad in gradients.items():
        print(f'Gradient for {name}:\n{grad}')

def test_linear():
    print("=== test_linear")
    x = torch.tensor([
        [1.0,2,3],
        [4,5,6],
    ], requires_grad=True)
    print("x", x)

    model = torch.nn.Linear(3, 4)

    params = model.state_dict()
    params["weight"][:,:] = torch.tensor(
      [[ 0.5210, -0.3797,  0.2674],
      [-0.5357, -0.1399,  0.0647],
      [ 0.3203,  0.0407, -0.3343],
      [ 0.2107, -0.1692,  0.5243]]
    )
    params["bias"][:] = torch.tensor(
        [ 0.3992,  0.3767,  0.5552, -0.2610]
    )

    # Collect params
    for name, param in model.named_parameters():
        print(name)
        print(param)

    y = model(x)
    print("y", y)


def test_encoder_layer():
    print("=== test_encoder_layer")
    emb_size = 4
    num_heads = 2
    ff_size = 3
    dropout = 0.0
    
    model = transformer.EncoderLayer(emb_size, num_heads, ff_size, dropout)
    params = model.state_dict()

    x = torch.tensor(
        [[[1,2,3,4],
          [5,6,7,8],
          [0.0878, 0.0416, 0.6166, 0.1477],
          [-0.3883,  0.2742, -0.4652, -0.1417],
          [0.5300, 0.2800, 0.5306, 0.4950]]],
         requires_grad=True
    )
    print("x", x.shape, x)
    
    # attention mask on q,k.T product
    mask = torch.ones((x.shape[0],x.shape[1],x.shape[1])).bool()
    print("mask", mask.shape, mask)

    params = model.state_dict()
    params["self_attn.W_q.weight"][:,:] = torch.tensor(
            [[-1.2321, -0.4785, -0.4598, -0.1860],
            [ 0.4576,  0.4961, -0.0903, -0.4833],
            [-0.1442,  0.3495,  0.4236, -0.0846],
            [-0.3082,  0.0956, -0.2470,  0.3061]])
    params["self_attn.W_q.bias"][:] = torch.tensor(
            [-1.3717, -0.1179, -0.0096, -0.4240])
    params["self_attn.W_k.weight"][:,:] = torch.tensor(
           [[-2.2321, -0.4785, -0.4598, -0.1860],
            [ 0.4576,  0.4961, -0.0903, -0.4833],
            [-0.1442,  0.3495,  0.4236, -0.0846],
            [-0.3082,  0.0956, -0.2470,  0.3061]])
    params["self_attn.W_k.bias"][:] = torch.tensor(
           [-2.3717, -0.1179, -0.0096, -0.4240])
    params["self_attn.W_v.weight"][:,:] = torch.tensor(
           [[-3.2321, -0.4785, -0.4598, -0.1860],
            [ 0.4576,  0.4961, -0.0903, -0.4833],
            [-0.1442,  0.3495,  0.4236, -0.0846],
            [-0.3082,  0.0956, -0.2470,  0.3061]])
    params["self_attn.W_v.bias"][:] = torch.tensor(
           [-3.3717, -0.1179, -0.0096, -0.4240])
    params["self_attn.W_o.weight"][:,:] = torch.tensor(
           [[-4.2321, -0.4785, -0.4598, -0.1860],
            [ 0.4576,  0.4961, -0.0903, -0.4833],
            [-0.1442,  0.3495,  0.4236, -0.0846],
            [-0.3082,  0.0956, -0.2470,  0.3061]])
    params["self_attn.W_o.bias"][:] = torch.tensor(
           [-4.3717, -0.1179, -0.0096, -0.4240])
    params["feed_forward.w_1.weight"][:,:] = torch.tensor(
           [[-5.4208,  0.2836, -0.1770,  0.3684],
            [ 0.3448,  0.4124, -0.2545,  0.2874],
            [-0.4372,  0.4165, -0.2362,  0.1144]])
    params["feed_forward.w_1.bias"][:] = torch.tensor(
            [ 5.2621, -0.3262,  0.4815])
    params["feed_forward.w_2.weight"][:,:] = torch.tensor(
            [[-6.3926, -0.1717,  0.2300],
            [ 0.0701,  0.3166, -0.2458],
            [ 0.1431, -0.3391,  0.5407],
            [ 0.4126, -0.3719,  0.5352]])
    params["feed_forward.w_2.bias"][:] = torch.tensor(
          [-6.5333, -0.0515, -0.1337,  0.0297])
    """
    params["norm1.weight"][:] = torch.tensor(
          [1., 1., 1., 1.])
    params["norm1.bias"][:] = torch.tensor(
          [0., 0., 0., 0.])
    params["norm2.weight"][:] = torch.tensor(
          [1., 1., 1., 1.])
    params["norm2.bias"][:] = torch.tensor(
          [0., 0., 0., 0.])
    """

    print("Start Params")
    for name in params:
        print("name:", name)
        param = params[name]
        print("shape:", param.shape)
        print(param)
    print("End Params")

    A = model(x, mask)
    print("output")
    print(A)

    dA = torch.ones_like(A)
    dA[0,0,0] = 1250
    print("dA", dA)

    A.backward(dA)
    print("dAdx", x.grad)

    # Collect gradients
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
          gradients[name] = param.grad.clone()

    # Print or use the gradients as needed
    for name, grad in gradients.items():
        print(f'Gradient for {name}:\n{grad}')


def test_decoder_layer():
    print("=== test_decoder_layer")
    emb_size = 4
    num_heads = 2
    ff_size = 3
    dropout = 0.0
    
    model = transformer.DecoderLayer(emb_size, num_heads, ff_size, dropout)
    params = model.state_dict()

    e = torch.tensor([[
        [-1.7227,  0.4192,  0.5928,  0.7106],
        [-1.7228,  0.4187,  0.5948,  0.7092],
        [-1.7241,  0.4332,  0.5882,  0.7027],
        [-1.7244,  0.4406,  0.5782,  0.7056],
        [-1.7241,  0.4344,  0.5848,  0.7048],
        ]], requires_grad=True)
    
    x = torch.tensor([[
        [0.1878, 0.5416, -0.1166, 0.4477],
        [0.2878, -0.6416, 0.2166, -0.9477],
        [0.0878, 0.0416, 0.6166, 0.1477],
        [-0.3883,  0.2742, -0.4652, -0.1417],
        [0.5300, 0.2800, 0.5306, 0.4950],
        ]], requires_grad=True)
    print("e", e.shape, e)
    print("x", x.shape, x)
    
    # attention masks
    src_mask = torch.ones((x.shape[0],x.shape[1],x.shape[1])).bool()
    tgt_mask = torch.tensor([[
        [1,0,0,0,0],
        [1,1,0,0,0],
        [1,1,1,0,0],
        [1,1,1,1,0],
        [1,1,1,1,1],
        ]]).bool()
    print("src_mask", src_mask.shape, src_mask)
    print("tgt_mask", tgt_mask.shape, tgt_mask)

    params = model.state_dict()
    params["self_attn.W_q.weight"][:,:] = torch.tensor(
            [[-1.2321, -0.4785, -0.4598, -0.1860],
            [ 0.4576,  0.4961, -0.0903, -0.4833],
            [-0.1442,  0.3495,  0.4236, -0.0846],
            [-0.3082,  0.0956, -0.2470,  0.3061]])
    params["self_attn.W_q.bias"][:] = torch.tensor(
            [-1.3717, -0.1179, -0.0096, -0.4240])
    params["self_attn.W_k.weight"][:,:] = torch.tensor(
           [[-2.2321, -0.4785, -0.4598, -0.1860],
            [ 0.4576,  0.4961, -0.0903, -0.4833],
            [-0.1442,  0.3495,  0.4236, -0.0846],
            [-0.3082,  0.0956, -0.2470,  0.3061]])
    params["self_attn.W_k.bias"][:] = torch.tensor(
           [-2.3717, -0.1179, -0.0096, -0.4240])
    params["self_attn.W_v.weight"][:,:] = torch.tensor(
           [[-3.2321, -0.4785, -0.4598, -0.1860],
            [ 0.4576,  0.4961, -0.0903, -0.4833],
            [-0.1442,  0.3495,  0.4236, -0.0846],
            [-0.3082,  0.0956, -0.2470,  0.3061]])
    params["self_attn.W_v.bias"][:] = torch.tensor(
           [-3.3717, -0.1179, -0.0096, -0.4240])
    params["self_attn.W_o.weight"][:,:] = torch.tensor(
           [[-4.2321, -0.4785, -0.4598, -0.1860],
            [ 0.4576,  0.4961, -0.0903, -0.4833],
            [-0.1442,  0.3495,  0.4236, -0.0846],
            [-0.3082,  0.0956, -0.2470,  0.3061]])
    params["self_attn.W_o.bias"][:] = torch.tensor(
           [-4.3717, -0.1179, -0.0096, -0.4240])
        
    params["cross_attn.W_q.weight"][:,:] = torch.tensor(
       [[ 0.0675,  0.0034,  0.2860, -0.0438],
        [ 0.3234,  0.4208, -0.0814, -0.0883],
        [-0.3376,  0.2880,  0.0641, -0.4295],
        [ 0.4480,  0.4328, -0.4657,  0.1207]])
    params["cross_attn.W_q.bias"][:] = torch.tensor(
        [-0.3390,  0.0716,  0.4804, -0.4253])
    params["cross_attn.W_k.weight"][:,:] = torch.tensor(
       [[ 0.2975,  0.0247,  0.4618, -0.1429],
        [-0.0016, -0.0542, -0.3919,  0.1051],
        [ 0.4285,  0.0760, -0.3002, -0.2579],
        [-0.1038,  0.4511,  0.4412,  0.2605]])
    params["cross_attn.W_k.bias"][:] = torch.tensor(
        [-0.3793,  0.4552,  0.1502,  0.3554])
    params["cross_attn.W_v.weight"][:,:] = torch.tensor(
       [[-0.4192, -0.4004,  0.0120, -0.4717],
        [-0.3308, -0.4728, -0.1381,  0.3374],
        [ 0.1521, -0.1548,  0.2885,  0.4352],
        [-0.1196, -0.2579, -0.3167,  0.0128]])
    params["cross_attn.W_v.bias"][:] = torch.tensor(
        [0.4992, -0.2558,  0.1871, -0.3701])
    params["cross_attn.W_o.weight"][:,:] = torch.tensor(
       [[ 1.5146e-01,  5.0816e-02,  3.9053e-04, -4.6405e-01],
        [-1.2832e-01, -4.3910e-01, -1.8390e-01, -5.1324e-02],
        [ 4.4734e-01, -3.3816e-01,  1.3738e-01, -1.3041e-01],
        [ 1.8204e-01, -2.9708e-01,  3.2434e-01, -6.3109e-02]])
    params["cross_attn.W_o.bias"][:] = torch.tensor(
        [-0.4427, -0.0959, -0.2821, -0.2209])
        
    params["feed_forward.w_1.weight"][:,:] = torch.tensor(
           [[-5.4208,  0.2836, -0.1770,  0.3684],
            [ 0.3448,  0.4124, -0.2545,  0.2874],
            [-0.4372,  0.4165, -0.2362,  0.1144]])
    params["feed_forward.w_1.bias"][:] = torch.tensor(
            [ 5.2621, -0.3262,  0.4815])
    params["feed_forward.w_2.weight"][:,:] = torch.tensor(
            [[-6.3926, -0.1717,  0.2300],
            [ 0.0701,  0.3166, -0.2458],
            [ 0.1431, -0.3391,  0.5407],
            [ 0.4126, -0.3719,  0.5352]])
    params["feed_forward.w_2.bias"][:] = torch.tensor(
          [-6.5333, -0.0515, -0.1337,  0.0297])
    """
    params["norm1.weight"][:] = torch.tensor(
          [1., 1., 1., 1.])
    params["norm1.bias"][:] = torch.tensor(
          [0., 0., 0., 0.])
    params["norm2.weight"][:] = torch.tensor(
          [1., 1., 1., 1.])
    params["norm2.bias"][:] = torch.tensor(
          [0., 0., 0., 0.])
    """

    print("Start Params")
    for name in params:
        print("name:", name)
        param = params[name]
        print("shape:", param.shape)
        print(param)
    print("End Params")

    A = model(x, e, src_mask, tgt_mask)
    print("output")
    print(A)

    dA = torch.ones_like(A)
    dA[0,:,0] = 1250
    print("dA", dA)

    A.backward(dA)
    print("dAdx", x.grad)
    print("dAde", e.grad)

    # Collect gradients
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
          gradients[name] = param.grad.clone()

    # Print or use the gradients as needed
    for name, grad in gradients.items():
        print(f'Gradient for {name}:\n{grad}')

def test_transformer():
    print("=== test_transformer")
    src_vocab_size = 10
    tgt_vocab_size = 10
    num_layers = 1
    emb_size = 4
    num_heads = 2
    max_seq_length = 5
    ff_size = 3
    dropout = 0.0
    sos_token = 8
    eos_token = 9
    pad_token = 0
    
    model = transformer.Transformer(src_vocab_size, tgt_vocab_size,
      emb_size, num_heads, num_layers, ff_size, max_seq_length, dropout)
    params = model.state_dict()

    src = torch.tensor([
        [1, 2, 3, pad_token, pad_token],
        ], requires_grad=False)
    
    tgt = torch.tensor([
        [sos_token, 1, 2, 3, pad_token],
        ], requires_grad=False)
    print("src", src.shape, src)
    print("tgt", tgt.shape, tgt)

    params = model.state_dict()

    # embeddings
    params["encoder_embedding.weight"][:,:] = torch.tensor(
        [[-1.5104, -0.0284,  0.8700,  0.8670],
        [ 1.1990,  1.1361,  0.8644,  0.1473],
        [-0.7335,  0.3807,  0.7741,  0.4396],
        [-0.7043,  1.6892, -0.5124,  0.4657],
        [ 0.0162, -1.9550,  1.6194,  1.5560],
        [ 1.3285,  0.2094, -1.5481,  0.2234],
        [-0.8587, -0.4757,  0.1260, -1.9552],
        [-0.7352, -0.9533, -0.7015, -1.8978],
        [-0.3166,  1.9976,  0.1297,  0.9044],
        [ 0.7586, -1.0734, -0.4338, -0.8578]])    
    params["decoder_embedding.weight"][:,:] = torch.tensor(
        [[-0.5281,  0.1697, -0.9366,  0.0129],
        [ 0.1969, -1.1860,  0.0961, -0.6315],
        [-0.1603, -0.1080,  0.1573,  0.9020],
        [ 1.2745,  0.0302,  1.0822, -0.4542],
        [-0.5246, -0.9769, -0.6335,  2.4609],
        [-0.7565,  1.0006, -1.5738,  1.5421],
        [-0.7897, -3.4246,  0.4188, -1.1293],
        [ 0.9453, -0.1520, -0.7238,  0.7383],
        [ 0.0729,  0.3859,  0.0163, -0.9825],
        [-0.9148,  1.0367, -1.0432,  1.4304]])

    # positional_encoding.pe is precomputed constant

    # encoder

    params["encoder_layers.0.self_attn.W_q.weight"][:,:] = torch.tensor(
            [[-1.2321, -0.4785, -0.4598, -0.1860],
            [ 0.4576,  0.4961, -0.0903, -0.4833],
            [-0.1442,  0.3495,  0.4236, -0.0846],
            [-0.3082,  0.0956, -0.2470,  0.3061]])
    params["encoder_layers.0.self_attn.W_q.bias"][:] = torch.tensor(
            [-1.3717, -0.1179, -0.0096, -0.4240])
    params["encoder_layers.0.self_attn.W_k.weight"][:,:] = torch.tensor(
           [[-2.2321, -0.4785, -0.4598, -0.1860],
            [ 0.4576,  0.4961, -0.0903, -0.4833],
            [-0.1442,  0.3495,  0.4236, -0.0846],
            [-0.3082,  0.0956, -0.2470,  0.3061]])
    params["encoder_layers.0.self_attn.W_k.bias"][:] = torch.tensor(
           [-2.3717, -0.1179, -0.0096, -0.4240])
    params["encoder_layers.0.self_attn.W_v.weight"][:,:] = torch.tensor(
           [[-3.2321, -0.4785, -0.4598, -0.1860],
            [ 0.4576,  0.4961, -0.0903, -0.4833],
            [-0.1442,  0.3495,  0.4236, -0.0846],
            [-0.3082,  0.0956, -0.2470,  0.3061]])
    params["encoder_layers.0.self_attn.W_v.bias"][:] = torch.tensor(
           [-3.3717, -0.1179, -0.0096, -0.4240])
    params["encoder_layers.0.self_attn.W_o.weight"][:,:] = torch.tensor(
           [[-4.2321, -0.4785, -0.4598, -0.1860],
            [ 0.4576,  0.4961, -0.0903, -0.4833],
            [-0.1442,  0.3495,  0.4236, -0.0846],
            [-0.3082,  0.0956, -0.2470,  0.3061]])
    params["encoder_layers.0.self_attn.W_o.bias"][:] = torch.tensor(
           [-4.3717, -0.1179, -0.0096, -0.4240])
    params["encoder_layers.0.feed_forward.w_1.weight"][:,:] = torch.tensor(
           [[-5.4208,  0.2836, -0.1770,  0.3684],
            [ 0.3448,  0.4124, -0.2545,  0.2874],
            [-0.4372,  0.4165, -0.2362,  0.1144]])
    params["encoder_layers.0.feed_forward.w_1.bias"][:] = torch.tensor(
            [ 5.2621, -0.3262,  0.4815])
    params["encoder_layers.0.feed_forward.w_2.weight"][:,:] = torch.tensor(
            [[-6.3926, -0.1717,  0.2300],
            [ 0.0701,  0.3166, -0.2458],
            [ 0.1431, -0.3391,  0.5407],
            [ 0.4126, -0.3719,  0.5352]])
    params["encoder_layers.0.feed_forward.w_2.bias"][:] = torch.tensor(
          [-6.5333, -0.0515, -0.1337,  0.0297])

    # decoder

    params["decoder_layers.0.self_attn.W_q.weight"][:,:] = torch.tensor(
            [[-1.2321, -0.4785, -0.4598, -0.1860],
            [ 0.4576,  0.4961, -0.0903, -0.4833],
            [-0.1442,  0.3495,  0.4236, -0.0846],
            [-0.3082,  0.0956, -0.2470,  0.3061]])
    params["decoder_layers.0.self_attn.W_q.bias"][:] = torch.tensor(
            [-1.3717, -0.1179, -0.0096, -0.4240])
    params["decoder_layers.0.self_attn.W_k.weight"][:,:] = torch.tensor(
           [[-2.2321, -0.4785, -0.4598, -0.1860],
            [ 0.4576,  0.4961, -0.0903, -0.4833],
            [-0.1442,  0.3495,  0.4236, -0.0846],
            [-0.3082,  0.0956, -0.2470,  0.3061]])
    params["decoder_layers.0.self_attn.W_k.bias"][:] = torch.tensor(
           [-2.3717, -0.1179, -0.0096, -0.4240])
    params["decoder_layers.0.self_attn.W_v.weight"][:,:] = torch.tensor(
           [[-3.2321, -0.4785, -0.4598, -0.1860],
            [ 0.4576,  0.4961, -0.0903, -0.4833],
            [-0.1442,  0.3495,  0.4236, -0.0846],
            [-0.3082,  0.0956, -0.2470,  0.3061]])
    params["decoder_layers.0.self_attn.W_v.bias"][:] = torch.tensor(
           [-3.3717, -0.1179, -0.0096, -0.4240])
    params["decoder_layers.0.self_attn.W_o.weight"][:,:] = torch.tensor(
           [[-4.2321, -0.4785, -0.4598, -0.1860],
            [ 0.4576,  0.4961, -0.0903, -0.4833],
            [-0.1442,  0.3495,  0.4236, -0.0846],
            [-0.3082,  0.0956, -0.2470,  0.3061]])
    params["decoder_layers.0.self_attn.W_o.bias"][:] = torch.tensor(
           [-4.3717, -0.1179, -0.0096, -0.4240])
        
    params["decoder_layers.0.cross_attn.W_q.weight"][:,:] = torch.tensor(
       [[ 0.0675,  0.0034,  0.2860, -0.0438],
        [ 0.3234,  0.4208, -0.0814, -0.0883],
        [-0.3376,  0.2880,  0.0641, -0.4295],
        [ 0.4480,  0.4328, -0.4657,  0.1207]])
    params["decoder_layers.0.cross_attn.W_q.bias"][:] = torch.tensor(
        [-0.3390,  0.0716,  0.4804, -0.4253])
    params["decoder_layers.0.cross_attn.W_k.weight"][:,:] = torch.tensor(
       [[ 0.2975,  0.0247,  0.4618, -0.1429],
        [-0.0016, -0.0542, -0.3919,  0.1051],
        [ 0.4285,  0.0760, -0.3002, -0.2579],
        [-0.1038,  0.4511,  0.4412,  0.2605]])
    params["decoder_layers.0.cross_attn.W_k.bias"][:] = torch.tensor(
        [-0.3793,  0.4552,  0.1502,  0.3554])
    params["decoder_layers.0.cross_attn.W_v.weight"][:,:] = torch.tensor(
       [[-0.4192, -0.4004,  0.0120, -0.4717],
        [-0.3308, -0.4728, -0.1381,  0.3374],
        [ 0.1521, -0.1548,  0.2885,  0.4352],
        [-0.1196, -0.2579, -0.3167,  0.0128]])
    params["decoder_layers.0.cross_attn.W_v.bias"][:] = torch.tensor(
        [0.4992, -0.2558,  0.1871, -0.3701])
    params["decoder_layers.0.cross_attn.W_o.weight"][:,:] = torch.tensor(
       [[ 1.5146e-01,  5.0816e-02,  3.9053e-04, -4.6405e-01],
        [-1.2832e-01, -4.3910e-01, -1.8390e-01, -5.1324e-02],
        [ 4.4734e-01, -3.3816e-01,  1.3738e-01, -1.3041e-01],
        [ 1.8204e-01, -2.9708e-01,  3.2434e-01, -6.3109e-02]])
    params["decoder_layers.0.cross_attn.W_o.bias"][:] = torch.tensor(
        [-0.4427, -0.0959, -0.2821, -0.2209])
        
    params["decoder_layers.0.feed_forward.w_1.weight"][:,:] = torch.tensor(
           [[-5.4208,  0.2836, -0.1770,  0.3684],
            [ 0.3448,  0.4124, -0.2545,  0.2874],
            [-0.4372,  0.4165, -0.2362,  0.1144]])
    params["decoder_layers.0.feed_forward.w_1.bias"][:] = torch.tensor(
            [ 5.2621, -0.3262,  0.4815])
    params["decoder_layers.0.feed_forward.w_2.weight"][:,:] = torch.tensor(
            [[-6.3926, -0.1717,  0.2300],
            [ 0.0701,  0.3166, -0.2458],
            [ 0.1431, -0.3391,  0.5407],
            [ 0.4126, -0.3719,  0.5352]])
    params["decoder_layers.0.feed_forward.w_2.bias"][:] = torch.tensor(
          [-6.5333, -0.0515, -0.1337,  0.0297])
    """
    params["norm1.weight"][:] = torch.tensor(
          [1., 1., 1., 1.])
    params["norm1.bias"][:] = torch.tensor(
          [0., 0., 0., 0.])
    params["norm2.weight"][:] = torch.tensor(
          [1., 1., 1., 1.])
    params["norm2.bias"][:] = torch.tensor(
          [0., 0., 0., 0.])
    """
    
    # fc

    params["fc.weight"][:,:] = torch.tensor(
        [[ 0.4024,  0.2209, -0.3322, -0.2039],
        [-0.0586, -0.3453, -0.4044,  0.3376],
        [-0.4428,  0.0175, -0.4929, -0.2737],
        [-0.4433, -0.2716, -0.0390,  0.4631],
        [-0.0599,  0.1389,  0.0554, -0.2265],
        [-0.4810, -0.2936,  0.2530, -0.0608],
        [ 0.1361,  0.1135, -0.1584, -0.0923],
        [ 0.3696, -0.2719, -0.0755,  0.3822],
        [-0.2697,  0.1172, -0.0242,  0.4085],
        [-0.2495, -0.1300,  0.2470,  0.3172]])
    params["fc.bias"][:] = torch.tensor(
        [ 0.1317, -0.1626, -0.0434, -0.4033,  0.0458, -0.1930,  0.3019, -0.3306,
        -0.1221,  0.3670])

    print("Start Params")
    for name in params:
        print("name:", name)
        param = params[name]
        print("shape:", param.shape)
        #print(param)
    print("End Params")

    A = model(src, tgt)
    print("output")
    print(A)

    """
    dA = torch.ones_like(A)
    dA[0,:,0] = 1250
    print("dA", dA)

    A.backward(dA)

    # Collect gradients
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
          gradients[name] = param.grad.clone()

    # Print or use the gradients as needed
    for name, grad in gradients.items():
        print(f'Gradient for {name}:\n{grad}')
    """

if __name__ == "__main__":
    #test_softmax()
    #test_scaled_dot_product_attention()
    #test_multihead_attention()
    #test_my_multihead_attention()
    #test_position_wise_feed_forward()
    #test_positional_encoding()
    #test_encoder_layer()
    #test_decoder_layer()
    test_transformer()
    #test_layernorm()
    #test_grad()
    #test_linear()
