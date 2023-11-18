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
    std::cout << fname << std::endl;
    std::cout << _graph.function(fname)->forward() << std::endl;
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

    // get sequance lenght - S, and embedding dimention - D
    int S = _k().rows();
    int D = _k().cols();

    // get qk_T attention component
    _attention = _graph.new_product(_q, *_graph.new_transpose(_k));

    // attention bias
    _bias = _graph.new_constant(S, S);

    // scale and add attention mask as bias
    _attention = &(*_attention / sqrt(D) + *_bias);
    _graph.name(_attention, "A before softmax");

    // apply softmax on qk_T rows, join results as columns
    Function* softmax = nullptr;
    for (int r=0; r<S; r++)
    {
      auto softmax_row = _graph.new_softmax(*_graph.new_split(*_attention, r,0, 1,S));
      if (softmax)
      {
        softmax = _graph.new_join(*softmax, *softmax_row, S,r+1); // shape as columns
      }
      else
      {
        softmax = softmax_row;
      }
    }

    // transpose joined softmax columns back to rows
    _attention = _graph.new_transpose(*softmax);
    _graph.name(_attention, "A after softmax");

    // apply dropout if present
    if (_dropout > 0)
    {
      _attention = _graph.new_dropout(*_attention, _dropout);
    }

    // complete qkv attention
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
  DTYPE _dropout;
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
