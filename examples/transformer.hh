/*
 * Copyright 2020-2023 Greg Padiasek. All Rights Reserved.
 */

#ifndef _TRANSFORMER_MODEL_H_
#define _TRANSFORMER_MODEL_H_

#include "main/graph.hh"


using namespace seegnify;


///////////////////////////////////
// SequenceMask
///////////////////////////////////
class SequenceMask : public Constant
{
public:
  SequenceMask(Graph& g,
  int tgt_tokens, int bos_token, int eos_token, int pad_token, int seq_size) :
  Constant(g),
  _tgt_tokens(tgt_tokens),
  _bos_token(bos_token),
  _eos_token(eos_token),
  _pad_token(pad_token),
  _seq_size(seq_size)
  {
  }

  void source(const std::vector<int>& src_seq)
  {
    _value = Tensor::Zero(_seq_size, _seq_size);

    auto bos = src_seq[0] == _bos_token;
    auto padding = _seq_size - sequence_size(src_seq) - bos;

    if (padding)
    {
      DTYPE inf = std::numeric_limits<DTYPE>::infinity();
      _value.rightCols(padding) = Tensor::Constant(_seq_size, padding, -inf);
    }
  }

  void target(const std::vector<int>& tgt_seq)
  {
    DTYPE inf = std::numeric_limits<DTYPE>::infinity();
    _value = Tensor::Constant(_seq_size, _seq_size, -inf);
    _value.triangularView<Eigen::Lower>().setConstant(0);

    auto bos = tgt_seq[0] == _bos_token;
    auto padding = _seq_size - sequence_size(tgt_seq) - bos;

    if (padding)
    {
      _value.rightCols(padding) = Tensor::Constant(_seq_size, padding, -inf);
    }
  }

  void output(const std::vector<int>& tgt_seq)
  {
    _value = Tensor::Zero(_seq_size, _tgt_tokens);

    if (!tgt_seq.size()) return;

    auto bos = tgt_seq[0] == _bos_token;
    auto tgt_size = sequence_size(tgt_seq);

    // one-hot token
    for (int i=0; i<tgt_size; i++)
    {
      _value(i, (int)tgt_seq[i+bos]) = 1;
    }

    // one-hot EOS
    if (tgt_size < _seq_size)
    {
      _value(tgt_size, _eos_token) = 1;
    }
  }

protected:

  // determine actual sequence size
  int sequence_size(const std::vector<int>& sequence)
  {
    // regular token count
    int count = 0;

    // determine size of the source sequence
    for (auto e: sequence)
    {
      if (e == _bos_token) continue;
      if (e == _eos_token) continue;
      if (e == _pad_token) continue;

      count++;
    }

    // limit seqence size to max sequence size
    return std::min(count, _seq_size);
  }

private:
  const int _tgt_tokens;
  const int _bos_token;
  const int _eos_token;
  const int _pad_token;
  const int _seq_size;
};


///////////////////////////////////
// Scaled Dot-Product Attention
///////////////////////////////////
class ScaledDotProductAttention : public Function
{
public:
  // q - query vectors
  // k - key vectors
  // v - value vectors
  // trg_size - target and query size
  // seq_size - max sequence size
  // head_size - head size
  // mask - attention mask [trg_size x seq_size] (dim of Q x K.T)
  // dropout - dropout probability
  ScaledDotProductAttention(
    Graph& g, Function& q, Function& k, Function& v, Function& mask, 
    int trg_size, int seq_size, int head_size, DTYPE dropout=0.0
  ) :
  Function(g), _q(q), _k(k), _v(v), _dropout(dropout)
  {
    // L - target lenght, S - sequance lenght, D - embedding dimension
    int L = trg_size;  // _q().rows()
    int S = seq_size;  // _k().rows()
    int D = head_size; // _k().cols()

    // get qk_T attention component [LxS]
    _attention = _graph.new_product(_q, *_graph.new_transpose(_k));

    // scale and add attention mask as bias [LxS]
    _attention = &(*_attention / sqrt(D) + mask);

    // apply row-wise softmax [LxS]
    _attention = _graph.new_rowwise(*_attention, L, S, [&](Function& row) {
      return _graph.new_softmax(row);
    });

    // apply dropout if present
    if (_dropout > 0)
    {
      _attention = _graph.new_dropout(*_attention, _dropout);
    }

    // complete qkv attention []
    _attention = _graph.new_product(*_attention, _v);

    _attention->derivative(_graph.new_iderivative(*this));
  }

  virtual const Tensor& forward()
  {
    if (_value.size() > 0) return _value;

    _value = _attention->forward();

    return _value;
  }
  
protected:
  Function& _q;
  Function& _k;
  Function& _v;
  Function* _attention;
  const DTYPE _dropout;
};


///////////////////////////////////
// Multi-Head Attention
///////////////////////////////////
class MultiHeadAttention : public Function
{
public:
  MultiHeadAttention(
    Graph& g, Function& q, Function& k, Function& v, Function& mask,
    int trg_size, int seq_size, int emb_size, int num_heads,
    bool bias = true, DTYPE dropout = 0.0) :
    Function(g)
  {
    const int L = trg_size; // q.rows()
    const int S = seq_size; // k.rows()
    const int E = emb_size; // v.cols()
    const int H = num_heads;
    const int D = emb_size / num_heads;

    auto q_heads = split_heads(linear(q, E, E, bias, "Wq", "bq"), H, S, D);
    auto k_heads = split_heads(linear(k, E, E, bias, "Wk", "bk"), H, S, D);
    auto v_heads = split_heads(linear(v, E, E, bias, "Wv", "bv"), H, S, D);

    std::vector<Function*> heads(num_heads, nullptr);

    for (int i=0; i<num_heads; i++)
    {
      heads[i] = new ScaledDotProductAttention(
        _graph, *q_heads[i], *k_heads[i], *v_heads[i], mask, L, S, H, dropout
      );
      _graph.keep(heads[i]);
    }

    auto& joined = join_heads(heads, S, H);
    _attention = &linear(joined, E, E, bias, "Wo", "bo");

    _attention->derivative(_graph.new_iderivative(*this));
  }

  virtual const Tensor& forward()
  {
    if (_value.size() > 0) return _value;

    // TODO: run attention heads in parallel using thread pool

    _value = _attention->forward();

    return _value;
  }
  
private:
  Function& linear(Function& x, int in, int out, bool bias,
  const std::string& w_name, const std::string& b_name)
  {
    auto W = _graph.new_variable(out, in, ("MHA." + w_name).c_str());
    auto y = _graph.new_product(x, *_graph.new_transpose(*W));

    if (bias)
    {
      auto b = _graph.new_variable(1, out, ("MHA." + b_name).c_str());
      return *y + *_graph.new_broadcast(*b, *y);
    }
    else
    {
      return *y;
    }
  }

  std::vector<Function*> split_heads(
    Function& x, int num_heads, int seq_size, int head_size
  )
  {
    std::vector<Function*> heads(num_heads, nullptr);

    // sequence length, head size
    int S = seq_size;
    int D = head_size;

    // split horizontally
    for (int i=0; i<num_heads; i++)
    {
      heads[i] = _graph.new_split(x, 0,i*D, S,D);
    }

    return heads;
  }

  Function& join_heads(
    const std::vector<Function*>& heads, int seq_size, int head_size
  )
  {
    // sequence length, head size
    int S = seq_size;
    int D = head_size;

    int num_heads = heads.size();

    Function* joined = nullptr;

    // join horizontally
    for (int i=0; i<num_heads; i++)
    {
      // transpose for row-major join
      auto head = _graph.new_transpose(*heads[i]);

      if (joined)
      {
        joined = _graph.new_join(*joined, *head, (i+1)*D,S);
      }
      else
      {
        joined = head;
      }
    }

    // inverse the transpose
    return *_graph.new_transpose(*joined);
  }

private:
  Function *_attention;
};


///////////////////////////////////
// PositionWiseFeedForward
///////////////////////////////////
class PositionWiseFeedForward : public Function
{
public:
  PositionWiseFeedForward(
    Graph& g, Function& x, int emb_size, int ff_size, DTYPE dropout = 0.0) :
    Function(g)
  {
    _y = g.new_linear(x, emb_size, ff_size, true);
    _y = g.new_relu(*_y);
    if (dropout > 0)
    {
      _y = g.new_dropout(*_y, dropout);
    }
    _y = g.new_linear(*_y, ff_size, emb_size, true);

    _y->derivative(g.new_iderivative(*this));
  }

  virtual const Tensor& forward()
  {
    if (_value.size() > 0) return _value;

    // update value
    _value = _y->forward();

    return _value;
  }

private:
  Function* _y;
};


///////////////////////////////////
// PositionalEncoding
///////////////////////////////////
class PositionalEncoding : public Function
{
public:
  PositionalEncoding(
    Graph& g, Function& x, int max_seq_size, int emb_size) :
    Function(g), _x(x)
  {
    Tensor position(max_seq_size, 1);
    Tensor div_term(1, emb_size / 2);

    double term = -(log(10000.0) / emb_size);

    for (int i=0; i<max_seq_size; i++) position(i) = i;
    for (int i=0; i<emb_size; i+=2) div_term(i/2) = exp(i * term);

    Tensor prod = position * div_term;
    _pe.resize(max_seq_size, emb_size);

    _pe(Eigen::all, Eigen::seq(0, emb_size-1, 2)) = prod.array().sin();
    _pe(Eigen::all, Eigen::seq(1, emb_size-1, 2)) = prod.array().cos();

    _x.derivative(g.new_iderivative(*this));
  }

  virtual const Tensor& forward()
  {
    if (_value.size() > 0) return _value;

    auto& x = _x.forward();
    int seq_size = x.rows();

    // update value
    _value = x + _pe(Eigen::seqN(0, seq_size), Eigen::all);

    return _value;
  }

private:
  Tensor _pe;
  Function& _x;
};

///////////////////////////////////
// RowwiseNorm
///////////////////////////////////
class RowwiseNorm : public Function
{
public:
  RowwiseNorm(Graph& g, Function& x, int rows, int cols, DTYPE eps = EPSILON) :
  Function(g)
  {
    // apply row-wise Norm with shared weights
    _y = _graph.new_rowwise(x, rows, cols, 
      [&](Function& row)
      {
        return _graph.new_norm(row, 1, cols, eps);
      },
      [&](Function& row, Function& shared)
      {
        return _graph.new_norm(row, (Norm&)shared);
      }
    );

    _y->derivative(g.new_iderivative(*this));
  }

  virtual const Tensor& forward()
  {
    if (_value.size() > 0) return _value;

    // update value
    _value = _y->forward();

    return _value;
  }

private:
  Function* _y;
};

///////////////////////////////////
// EncoderLayer
///////////////////////////////////
class EncoderLayer : public Function
{
public:
  EncoderLayer(
    Graph& g, Function& x, Function& mask,
    int seq_size, int emb_size, int num_heads, int ff_size, DTYPE dropout) :
    Function(g)
  {
    auto attn = new MultiHeadAttention(g, x, x, x, mask,
      seq_size, seq_size, emb_size, num_heads, true, dropout);
    g.keep(attn);

    _y = new RowwiseNorm(
      g, x + *g.new_dropout(*attn, dropout), seq_size, emb_size);
    g.keep(_y);

    auto ff = new PositionWiseFeedForward(g, *_y, emb_size, ff_size, dropout);
    g.keep(ff);

    _y = new RowwiseNorm(
      g, *_y + *g.new_dropout(*ff, dropout), seq_size, emb_size);
    g.keep(_y);

    _y->derivative(g.new_iderivative(*this));
  }

  virtual const Tensor& forward()
  {
    if (_value.size() > 0) return _value;

    // update value
    _value = _y->forward();

    return _value;
  }

private:
  Function* _y;
};

///////////////////////////////////
// DecoderLayer
///////////////////////////////////
class DecoderLayer : public Function
{
public:
  DecoderLayer(
    Graph& g, Function& x, Function& e, Function& src_mask, Function& tgt_mask,
    int seq_size, int emb_size, int num_heads, int ff_size, DTYPE dropout) :
    Function(g)
  {
    auto self_attn = new MultiHeadAttention(g, x, x, x, tgt_mask,
      seq_size, seq_size, emb_size, num_heads, true, dropout);
    g.keep(self_attn);

    _y = new RowwiseNorm(
      g, x + *g.new_dropout(*self_attn, dropout), seq_size, emb_size);
    g.keep(_y);

    auto cross_attn = new MultiHeadAttention(g, *_y, e, e, src_mask,
      seq_size, seq_size, emb_size, num_heads, true, dropout);
    g.keep(cross_attn);

    _y = new RowwiseNorm(
      g, *_y + *g.new_dropout(*cross_attn, dropout), seq_size, emb_size);
    g.keep(_y);

    auto ff = new PositionWiseFeedForward(g, *_y, emb_size, ff_size, dropout);
    g.keep(ff);

    _y = new RowwiseNorm(
      g, *_y + *g.new_dropout(*ff, dropout), seq_size, emb_size);
    g.keep(_y);

    _y->derivative(g.new_iderivative(*this));
  }

  virtual const Tensor& forward()
  {
    if (_value.size() > 0) return _value;

    // update value
    _value = _y->forward();

    return _value;
  }

private:
  Function* _y;
};


///////////////////////////////////
// Transformer
///////////////////////////////////
class Transformer : public Function
{
public:
  Transformer(
  Graph& g, int src_tokens, int tgt_tokens,
  int bos_token, int eos_token, int pad_token,
  int num_layers, int num_heads, int emb_size, int ff_size, int seq_size,
  DTYPE dropout) : Function(g)
  {
    _src = g.new_constant(1, seq_size);
    _tgt = g.new_constant(1, seq_size);

    auto src_emb = g.new_embedding(*_src, src_tokens, emb_size);
    auto tgt_emb = g.new_embedding(*_tgt, tgt_tokens, emb_size);

    auto src_pos = new PositionalEncoding(g, *src_emb, seq_size, emb_size);
    auto tgt_pos = new PositionalEncoding(g, *tgt_emb, seq_size, emb_size);
    g.keep(src_pos);
    g.keep(tgt_pos);

    _src_mask = new SequenceMask(g,
      tgt_tokens, bos_token, eos_token, pad_token, seq_size);
    _tgt_mask = new SequenceMask(g,
      tgt_tokens, bos_token, eos_token, pad_token, seq_size);
    g.keep(_src_mask);
    g.keep(_tgt_mask);

    _encoder = g.new_dropout(*src_pos, dropout);
    _decoder = g.new_dropout(*tgt_pos, dropout);

    // encoder layers
    g.scope_push("encoder");
    for (int i=0; i<num_layers; i++)
    {
      _encoder = new EncoderLayer(g, *_encoder, *_src_mask,
        seq_size, emb_size, num_heads, ff_size, dropout);
      g.keep(_encoder);
    }
    g.scope_pop();

    // decoder layers
    g.scope_push("decoder");
    for (int i=0; i<num_layers; i++)
    {
      _decoder = new DecoderLayer(g, *_decoder, *_encoder, *_src_mask,
        *_tgt_mask, seq_size, emb_size, num_heads, ff_size, dropout);
      g.keep(_decoder);
    }
    g.scope_pop();

    _decoder = g.new_linear(*_decoder, emb_size, tgt_tokens);

    _decoder->derivative(g.new_iderivative(*this));
  }

  // requires src and tgt sequences
  virtual const Tensor& forward()
  {
    if (_value.size() > 0) return _value;

    // run transformer
    _value = _decoder->forward();

    return _value;
  }

  // forward with automatic setting of src and tgt sequences
  const Tensor& forward(
    const std::vector<int>& src, const std::vector<int>& tgt
  )
  {
    if (_value.size() > 0) return _value;

    // update source input tensor
    _src->value() = Tensor::Zero(1, src.size());
    for (int i=0; i<src.size(); i++) _src->value()(i) = src[i];

    // update target input tensor
    _tgt->value() = Tensor::Zero(1, tgt.size());
    for (int i=0; i<tgt.size(); i++) _tgt->value()(i) = tgt[i];

    // update source and terget attention masks
    _src_mask->source(src);
    _tgt_mask->target(tgt);

    return forward();
  }

  // source and target
  Function& source() { return *_src; }
  Function& target() { return *_tgt; }

  // access to encoder cached output for inferrence
  Function& encoder() { return *_encoder; }

private:
  Constant* _src;
  Constant* _tgt;
  SequenceMask* _src_mask;
  SequenceMask* _tgt_mask;
  Function* _encoder;
  Function* _decoder;
};


#endif /* _TRANSFORMER_MODEL_H_ */
