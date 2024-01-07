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
  SequenceMask(Graph& g, int max_seq_size) :
  Constant(g), _max_seq_size(max_seq_size)
  {
  }

  void source(int seq_size)
  {
    _value = Tensor::Zero(_max_seq_size, _max_seq_size);
    _value.leftCols(seq_size) = Tensor::Ones(_max_seq_size, seq_size);
  }

  void target(int seq_size)
  {
    _value = Tensor::Zero(_max_seq_size, _max_seq_size);
    _value.triangularView<Eigen::Lower>().setConstant(1);

    auto padding = std::max(_max_seq_size - seq_size, 0);

    if (padding)
    {
      _value.rightCols(padding) = Tensor::Zero(_max_seq_size, padding);
    }
  }

private:
  const int _max_seq_size;
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
  // mask - attention mask
  // dropout - dropout probability
  ScaledDotProductAttention(
    Graph& g, Function& q, Function& k, Function& v, Function* mask, 
    int trg_size, int seq_size, int head_size, DTYPE dropout=0.0
  ) :
  Function(g), _q(q), _k(k), _v(v), _mask(mask), _dropout(dropout)
  {
    // L - target lenght, S - sequance lenght, D - embedding dimension
    int L = trg_size;  // _q().rows()
    int S = seq_size;  // _k().rows()
    int D = head_size; // _k().cols()

    // get qk_T attention component [LxS]
    _attention = _graph.new_product(_q, *_graph.new_transpose(_k));

    // attention bias [LxS] (actual dimension updated in forward)
    _bias = _graph.new_constant();

    // scale and add attention mask as bias
    _attention = &(*_attention / sqrt(D) + *_bias);

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

    auto& q = _q();
    auto& k = _k();

    // get input/output dimensions
    int L = q.rows();
    int S = k.rows();

    // initialize default attention mask
    auto& b = _bias->value();
    b = Tensor::Zero(L, S);

    // set custom attention mask
    if (_mask)
    {
      auto& m = _mask->forward();
      DTYPE inf = std::numeric_limits<DTYPE>::infinity();
      b = (m.array() == 0).select(-inf, b);
    }

    _value = _attention->forward();

    return _value;
  }
  
protected:
  Function& _q;
  Function& _k;
  Function& _v;
  Function* _mask;
  Constant* _bias;
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
    Graph& g, Function& q, Function& k, Function& v, Function* mask,
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
    Graph& g, Function& x, Function* mask,
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
    Graph& g, Function& x, Function& e, Function* src_mask, Function* tgt_mask,
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
  Graph& g, Function& src_x, Function& tgt_x,
  int src_vocab_size, int tgt_vocab_size, int emb_size, int num_heads,
  int num_layers, int ff_size, int max_seq_size, DTYPE dropout) :
  Function(g), _src_x(src_x), _tgt_x(tgt_x)
  {
    auto src_emb = g.new_embedding(src_x, src_vocab_size, emb_size);
    auto tgt_emb = g.new_embedding(tgt_x, tgt_vocab_size, emb_size);

    auto src_pos = new PositionalEncoding(g, *src_emb, max_seq_size, emb_size);
    auto tgt_pos = new PositionalEncoding(g, *tgt_emb, max_seq_size, emb_size);
    g.keep(src_pos);
    g.keep(tgt_pos);

    _src_mask = new SequenceMask(g, max_seq_size);
    _tgt_mask = new SequenceMask(g, max_seq_size);
    g.keep(_src_mask);
    g.keep(_tgt_mask);

    Function* enc = g.new_dropout(*src_pos, dropout);
    Function* dec = g.new_dropout(*tgt_pos, dropout);

    // encoder layers
    for (int i=0; i<num_layers; i++)
    {
      enc = new EncoderLayer(g, *enc, _src_mask,
        max_seq_size, emb_size, num_heads, ff_size, dropout);
      g.keep(enc);
    }

    _encoder = enc;

    // decoder layers
    for (int i=0; i<num_layers; i++)
    {
      dec = new DecoderLayer(g, *dec, *enc, _src_mask, _tgt_mask,
        max_seq_size, emb_size, num_heads, ff_size, dropout);
      g.keep(dec);
    }

    _y = g.new_linear(*dec, emb_size, tgt_vocab_size);

    _y->derivative(g.new_iderivative(*this));
  }

  virtual const Tensor& forward()
  {
    if (_value.size() > 0) return _value;

    auto& src_x = _src_x();
    auto& tgt_x = _tgt_x();

    _src_mask->source(src_x.size());
    _tgt_mask->target(tgt_x.size());

    _value = _y->forward();

    return _value;
  }

  // access to encoder cached output for inferrence
  Function& encoder() { return *_encoder; }

protected:
  Function& _src_x;
  Function& _tgt_x;
  SequenceMask* _src_mask;
  SequenceMask* _tgt_mask;
  Function* _encoder;
  Function* _y;
};


#endif /* _TRANSFORMER_MODEL_H_ */
