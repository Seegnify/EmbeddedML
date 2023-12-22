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

    // attention bias [LxS]
    _bias = _graph.new_constant(L, S);

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
    int trg_size, int seq_size, int emb_size, int num_heads,
    bool bias = true, DTYPE dropout = 0.0) :
    Function(g)
  {
    const int L = trg_size; // q.rows()
    const int S = seq_size; // k.rows()
    const int E = emb_size; // v.cols()
    const int H = num_heads;
    const int D = emb_size / num_heads;

    // projection matrices
    _Wq = g.new_variable(E, E);
    _Wk = g.new_variable(E, E);
    _Wv = g.new_variable(E, E);
    _Wo = g.new_variable(E, E);

    // projection bias
    _bq = (bias) ? g.new_variable(1, E) : nullptr;
    _bk = (bias) ? g.new_variable(1, E) : nullptr;
    _bv = (bias) ? g.new_variable(1, E) : nullptr;
    _bo = (bias) ? g.new_variable(1, E) : nullptr;

    auto q_heads = split_heads(linear(q, _Wq, _bq), H, S, D);
    auto k_heads = split_heads(linear(k, _Wk, _bk), H, S, D);
    auto v_heads = split_heads(linear(v, _Wv, _bv), H, S, D);

    std::vector<Function*> heads(num_heads, nullptr);

    for (int i=0; i<num_heads; i++)
    {
      heads[i] = new ScaledDotProductAttention(
        _graph, *q_heads[i], *k_heads[i], *v_heads[i], nullptr, L, S, H, dropout
      );
      _graph.keep(heads[i]);
    }

    auto& joined = join_heads(heads, S, H);
    _attention = &linear(joined, _Wo, _bo);

    _attention->derivative(_graph.new_iderivative(*this));
  }

  virtual const Tensor& forward()
  {
    if (_value.size() > 0) return _value;

    // TODO: run attention heads in parallel using thread pool

    _value = _attention->forward();

    return _value;
  }
  
  // variable access
  Variable& Wq() { return *_Wq; }
  Variable& Wk() { return *_Wk; }
  Variable& Wv() { return *_Wv; }
  Variable& Wo() { return *_Wo; }
  
  Variable& bq() { return *_bq; }
  Variable& bk() { return *_bk; }
  Variable& bv() { return *_bv; }
  Variable& bo() { return *_bo; }

private:
  Function& linear(Function& x, Function* W, Function* b)
  {
    auto y = _graph.new_product(x, *_graph.new_transpose(*W));
    return (b) ? (*y + *_graph.new_broadcast(*b, *y)) : (*y);
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
  Variable *_Wq, *_Wk, *_Wv, *_Wo;
  Variable *_bq, *_bk, *_bv, *_bo;
  Function *_attention;
};


///////////////////////////////////
// PositionwiseFeedForward
///////////////////////////////////
class PositionwiseFeedForward : public Function
{
public:
  PositionwiseFeedForward(
    Graph& g, Function& x, int emb_size, int hid_size, DTYPE dropout = 0.0) :
    Function(g)
  {
    _l1 = g.new_linear(x, emb_size, hid_size);
    _y = g.new_relu(*_l1);
    if (dropout > 0)
    {
      _y = g.new_dropout(*_y, dropout);
    }
    _l2 = g.new_linear(*_y, hid_size, emb_size);
    _y = _l2;

    _y->derivative(g.new_iderivative(*this));
  }

  virtual const Tensor& forward()
  {
    if (_value.size() > 0) return _value;

    // update value
    _value = _y->forward();

    return _value;
  }

  // variable access
  Linear& L1() { return *_l1; }
  Linear& L2() { return *_l2; }

private:
  Linear *_l1;
  Linear* _l2;
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
    std::cout << "PositionalEncoding" << std::endl;
    Tensor position(max_seq_size, 1);
    Tensor div_term(1, emb_size / 2);

    double term = -(log(10000.0) / emb_size);
    std::cout << "term" << std::endl;
    std::cout << term << std::endl;

    for (int i=0; i<max_seq_size; i++) position(i) = i;
    for (int i=0; i<emb_size; i+=2) div_term(i/2) = exp(i * term);

    std::cout << "position" << std::endl;
    std::cout << position << std::endl;
    std::cout << "div_term" << std::endl;
    std::cout << div_term << std::endl;

    Tensor prod = position * div_term;
    std::cout << "position * div_term" << std::endl;
    std::cout << prod << std::endl;
    _pe.resize(max_seq_size, emb_size);

    _pe(Eigen::all, Eigen::seq(0, emb_size, 2)) = prod.array().sin();
    _pe(Eigen::all, Eigen::seq(1, emb_size, 2)) = prod.array().cos();
    std::cout << "_pe(Eigen::all, Eigen::seq(1, emb_size, 2))" << std::endl;
    std::cout << _pe << std::endl;

    _x.derivative(g.new_iderivative(*this));
  }

  virtual const Tensor& forward()
  {
    if (_value.size() > 0) return _value;

    auto& x = _x.forward();
    int seq_size = x.rows();

    std::cout << "PositionalEncoding::forward" << std::endl;
    std::cout << "seq_size=" << seq_size << std::endl;
    std::cout << "pe applied" << std::endl;
    std::cout << _pe(Eigen::seq(0, seq_size), Eigen::all) << seq_size << std::endl;

    // update value
    _value = x + _pe(Eigen::seqN(0, seq_size), Eigen::all);

    return _value;
  }

private:
  Tensor _pe;
  Function& _x;
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
