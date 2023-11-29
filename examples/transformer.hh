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
  // mask - attention mask
  // dropout - dropout probability
  Attention(
    Graph& g, Function& q, Function& k, Function& v,
    Function* mask=nullptr, DTYPE dropout=0.0
  ) :
  Function(g), _q(q), _k(k), _v(v), _mask(mask), _dropout(dropout)
  {
    // get scaled qkT
    _bias = nullptr;
    _attention = nullptr;
  }

  void print(const char* fname)
  {
    auto f = _graph.function(fname);
    if (f)
    {
      auto& t = f->forward();
      std::cout << fname << " [" << t.rows() << "x" << t.cols() << "]" << std::endl;
      std::cout << t << std::endl;
    }
    else
    {
      std::cout << fname << " [NOT FOUND]" << std::endl;
    }
  }

  virtual const Tensor& forward()
  {
    if (_value.size() > 0) return _value;

    init();

    // mask attention
    _bias->value() = Tensor::Zero(_bias->value().rows(), _bias->value().cols());

    if (_mask)
    {
      DTYPE inf = std::numeric_limits<DTYPE>::infinity();
      _bias->value() = (_mask->forward().array() == 0).select(-inf, _bias->value());
    }

    _value = _attention->forward();

    print("A before softmax");
    print("A after softmax");

    return _value;
  }

private:
  void init()
  {
    if (_attention) return;

    // L - target lenght, S - sequance lenght, D - embedding dimension
    int L = _q().rows();
    int S = _k().rows();
    int D = _k().cols();
    std::cout << "L=" << L << ", S=" << S << ", D=" << D << std::endl;

    // get qk_T attention component [LxS]
    _attention = _graph.new_product(_q, *_graph.new_transpose(_k));

    // attention bias [LxS]
    _bias = _graph.new_constant(L, S);

    // scale and add attention mask as bias, transpose for column softmax
    _attention = _graph.new_transpose(*_attention / sqrt(D) + *_bias);
    _graph.name(_attention, "A before softmax");

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
    _graph.name(_attention, "A after softmax");

    // apply dropout if present
    if (_dropout > 0)
    {
      _attention = _graph.new_dropout(*_attention, _dropout);
    }

    // complete qkv attention []
    _attention = _graph.new_product(*_attention, _v);

    _attention->derivative(_graph.new_iderivative(*this));
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
    int emb_size, int num_heads, DTYPE dropout=0.0) :
  Function(g), _q(q), _k(k), _v(v),
  _emb_size(emb_size), _num_heads(num_heads), _dropout(dropout)
  {
    _attention = nullptr;
  }

  virtual const Tensor& forward()
  {
    if (_value.size() > 0) return _value;

    init();

    _value = _attention->forward();

    return _value;
  }

private:

  std::vector<Function*> split_heads(Function& x)
  {
    std::vector<Function*> heads(_num_heads, nullptr);

    // sequence length, model size, embedding dimension
    int S = x().rows();
    int E = x().cols();
    int D = E / _num_heads;

    assert(E == _num_heads * D);

    for (int i=0; i<_num_heads; i++)
    {
      auto head = _graph.new_split(x, 0,i*D, S,D);
      heads[i] = _graph.new_transpose(*head);
    }

    return heads;
  }

  Function* join_heads(const std::vector<Function*>& heads)
  {
    Function* joined = nullptr;

    for (int i=0; i<_num_heads; i++)
    {
      auto head = _graph.new_transpose(*heads[i]);

      // sequence length, model size, embedding dimension
      int S = head->forward().cols();
      int D = head->forward().rows();

      if (joined)
      {
        joined = _graph.new_join(*joined, *head, S,(i+1)*D);
      }
      else
      {
        joined = head;
      }
    }

    return joined;
  }

  void init()
  {
    if (_attention) return;

    auto q_heads = split_heads(_q);
    auto k_heads = split_heads(_k);
    auto v_heads = split_heads(_v);

    std::vector<Function*> attn(_num_heads, nullptr);

    for (int i=0; i<_num_heads; i++)
    {
      attn[i] = new Attention(
        _graph, *q_heads[i], *k_heads[i], *v_heads[i], nullptr, _dropout
      );
      _graph.keep(attn[i]);
    }

    _attention = join_heads(attn);
  }

protected:
  Function& _q;
  Function& _k;
  Function& _v;
  Function* _attention;
  const int _emb_size;
  const int _num_heads;
  const DTYPE _dropout;
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
