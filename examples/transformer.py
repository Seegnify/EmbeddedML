#source: https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import numpy as np
import collections

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout=0.0, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    if attn_mask is not None:
        print("\nattn_weight:\n", attn_weight.shape)
        print("\nattn_mask:\n", attn_mask.shape)
        attn_weight.masked_fill(attn_mask.logical_not(), float("-inf"))
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout, train=True)
    output = attn_weight @ value
    return output

class SeqeunceMask:
    def __init__(self, src_pad_idx, trg_pad_idx):
      self.src_pad_idx = src_pad_idx
      self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):

        #src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_tgt_mask(self, trg):

        #trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        #trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).bool()

        #trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        #trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, name="MHA"):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            #print("attn_scores:\n", attn_scores.shape)
            #print("mask:\n", mask.shape)
            #print(self.name, "dot prod mask", mask)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
            #print(self.name, "dot prod mask attn_scores", attn_scores)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(self.dropout(attn_probs), V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        WQ = self.W_q(Q)
        WK = self.W_k(K)
        WV = self.W_v(V)
        Q = self.split_heads(WQ)
        K = self.split_heads(WK)
        V = self.split_heads(WV)
        
        attn_output_1 = self.scaled_dot_product_attention(Q, K, V, mask)
        #attn_output_2 = scaled_dot_product_attention(Q, K, V, mask)
        cobined_heads = self.combine_heads(attn_output_1)
        output = self.W_o(cobined_heads)
        return output

# source: https://nlp.seas.harvard.edu/2018/04/03/attention.html
class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        l1 = self.w_1(x)
        return self.w_2(self.dropout(torch.nn.functional.relu(l1)))


"""
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.relu(self.w_1(x)))
"""


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        term = -(math.log(10000.0) / d_model)
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * term)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
  

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x_d = x + self.dropout(attn_output)
        x = self.norm1(x_d)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, name="self_attn")
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, name="cross_attn")
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt, pad_token):
        src_mask = (src != pad_token).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != pad_token).unsqueeze(1).unsqueeze(2)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt, pad_token, enc_output = None):
        src_mask, tgt_mask = self.generate_mask(src, tgt, pad_token)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        if enc_output is None:
          enc_output = src_embedded
          for enc_layer in self.encoder_layers:
              enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output, enc_output

    def decode(self, output, row):
      row = output.detach().numpy()[row]
      token = np.argmax(row)
      return token

    def generate(self, src, bos_token, eos_token, pad_token):
      # set evaluation mode
      self.eval()

      # create source target sequences
      src = torch.tensor([src], requires_grad=False)
      tgt = torch.tensor([[pad_token] * src.size(1)], requires_grad=False)
      token = bos_token # initial target token

      output = []
      encoder_cache = None

      # generate next tokens
      for i in range(src.size(1)):
        tgt[0,i] = token
        y, encoder_cache = self.forward(src, tgt, pad_token, encoder_cache)
        token = self.decode(y[0], i)
        output.append(token)
        if token == eos_token:
          break

      # return generated sequence
      return output

def save_model_as_numpy(torch_state_dict, numpy_path):
  model_dict = collections.OrderedDict()
  for m in torch_state_dict:
    arr = torch_state_dict[m].numpy()
    model_dict[m] = arr
  np.savez(numpy_path, **model_dict)

def save_model_as_torch(numpy_state_dict, torch_path):
  model_dict = collections.OrderedDict()
  for m in numpy_state_dict:
    arr = numpy_state_dict[m]
    model_dict[m] = torch.FloatTensor(arr)
  torch.save(model_dict, torch_path)

def print_model(model_state_dict):
  for m in model_state_dict:
    arr = model_state_dict[m].numpy()
    print("tensor", m, arr.shape, arr.dtype)

def main():
  # Meta params
  src_vocab_size = 5000
  tgt_vocab_size = 5000
  d_model = 512
  num_heads = 8
  num_layers = 6
  d_ff = 2048
  max_seq_length = 100
  dropout = 0.1
  batch_size = 64
  num_epochs = 10
  model_path = "transformer.pt"
  model_np_path = "transformer.npz"
  train_data_path = "dataset.dat"

  transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

  # Load model
  try:
    transformer.load_state_dict(torch.load(model_path))
    print("Model loaded from", model_path)
  except:
    print("New model created")

  # Load data
  try:
    train_data = torch.load(train_data_path)
    src_data = train_data["src_data"]
    tgt_data = train_data["tgt_data"]
    print("Training data loaded from", train_data_path)
  except:
    # Generate random sample data
    src_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length))  # (batch_size, seq_length)
    tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length))  # (batch_size, seq_length)
    print("New training data generated")
  print("src_data", src_data.shape)
  print("tgt_data", tgt_data.shape, tgt_data[:, :-1].shape)

  # print_model(torch.load(model_path))

  # Model training
  criterion = nn.CrossEntropyLoss(ignore_index=0)
  optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

  transformer.train()

  for epoch in range(num_epochs):
      optimizer.zero_grad()

      # one-less target lenght
      #output = transformer(src_data, tgt_data[:, :-1])
      #loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))

      # same target lenght
      output = transformer(src_data, tgt_data)
      loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data.contiguous().view(-1))

      loss.backward()
      optimizer.step()
      print(f"Epoch: {epoch+1} / {num_epochs}, Loss: {loss.item()}")

  # Save data
  train_data = {"src_data":src_data, "tgt_data":tgt_data}
  torch.save(train_data, train_data_path)
  print("Training data saved to", train_data_path)

  # Save model
  torch.save(transformer.state_dict(), model_path)
  print("Torch model saved to", model_path)

  # Save model as numpy
  save_model_as_numpy(transformer.state_dict(), model_np_path)
  print("Model as numpy saved to", model_np_path)

if __name__ == "__main__":
  main()
