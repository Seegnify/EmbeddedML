/*
 * Copyright 2020-2023 Greg Padiasek. All Rights Reserved.
 */

#ifndef _TRANSFORMER_MODEL_H_
#define _TRANSFORMER_MODEL_H_

#include "main/graph.hh"
#include "main/storage.hh"


using namespace seegnify;


///////////////////////////////////
// Scaled Dot-Product Attention
///////////////////////////////////
class Attention : public Function
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
  Attention(
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

    // attention bias [LxS]
    _bias = _graph.new_constant(L, S);

    // scale and add attention mask as bias, transpose for column softmax
    _attention = _graph.new_transpose(*_attention / sqrt(D) + *_bias);

    // apply softmax on qk_T rows, join results as columns [SxL]
    Function* softmax = nullptr;
    for (int r=0; r<L; r++)
    {
      auto softmax_row = _graph.new_softmax(*_graph.new_split(*_attention, 0,r, S,1));
      if (softmax)
      {
        softmax = _graph.new_join(*softmax, *softmax_row, S,r+1); // shape as columns
      }
      else
      {
        softmax = softmax_row;
      }
    }

    // transpose joined softmax columns back to rows [LxS]
    _attention = _graph.new_transpose(*softmax);

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

    // mask attention
    _bias->value() = Tensor::Zero(_bias->value().rows(), _bias->value().cols());

    if (_mask)
    {
      DTYPE inf = std::numeric_limits<DTYPE>::infinity();
      _bias->value() = (_mask->forward().array() == 0).select(-inf, _bias->value());
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
    Graph& g, Function& q, Function& k, Function& v,
    int trg_size, int seq_size, int emb_size, int num_heads, DTYPE dropout=0.0) : 
    Function(g)
  {
    const int L = trg_size; // q.rows()
    const int S = seq_size; // k.rows()
    const int E = emb_size; // v.cols()
    const int H = num_heads;
    const int D = emb_size / num_heads;

    _Wq = g.new_variable(E, E);
    _Wk = g.new_variable(E, E);
    _Wv = g.new_variable(E, E);
    _Wo = g.new_variable(E, E);

    auto q_heads = split_heads(linear(q, *_Wq), H, S, D);
    auto k_heads = split_heads(linear(k, *_Wk), H, S, D);
    auto v_heads = split_heads(linear(v, *_Wv), H, S, D);

    std::vector<Function*> heads(num_heads, nullptr);

    for (int i=0; i<num_heads; i++)
    {
      heads[i] = new Attention(
        _graph, *q_heads[i], *k_heads[i], *v_heads[i], nullptr, L, S, H, dropout
      );
      _graph.keep(heads[i]);
    }

    auto& joined = join_heads(heads, S, H);
    auto& WoT = *_graph.new_transpose(*_Wo);
    _attention = _graph.new_product(joined, WoT);
  }

  virtual const Tensor& forward()
  {
    if (_value.size() > 0) return _value;

    // TODO: run attention heads in parallel using thread pool

    _value = _attention->forward();

    return _value;
  }

private:
  Function& linear(Function& x, Function& W)
  {
    return *_graph.new_product(x, *_graph.new_transpose(W));
  }

  std::vector<Function*> split_heads(
    Function& x, int num_heads, int seq_size, int head_size
  )
  {
    std::vector<Function*> heads(num_heads, nullptr);

    // sequence length, head size
    int S = seq_size;
    int D = head_size;

    for (int i=0; i<num_heads; i++)
    {
      auto head = _graph.new_split(x, 0,i*D, S,D);
      heads[i] = _graph.new_transpose(*head);
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

    for (int i=0; i<num_heads; i++)
    {
      auto head = _graph.new_transpose(*heads[i]);

      if (joined)
      {
        joined = _graph.new_join(*joined, *head, S,(i+1)*D);
      }
      else
      {
        joined = head;
      }
    }

    return *joined;
  }

protected:
  Function* _attention;
  Variable* _Wq;
  Variable* _Wk;
  Variable* _Wv;
  Variable* _Wo;
};


///////////////////////////////////
// Transformer
///////////////////////////////////
class Transformer : public Function
{
public:
  Transformer(Graph& g) : Function(g)
  {
  }

  virtual const Tensor& forward()
  {
    if (_value.size() > 0) return _value;

    return _value;
  }

protected:
};


#endif /* _TRANSFORMER_MODEL_H_ */
