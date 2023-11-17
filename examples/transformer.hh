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
  Attention(Graph& g, Function& q, Function& k, Function& v, Function* mask, DTYPE dropout) :
  Function(g), _q(q), _k(k), _v(v), _mask(mask), _dropout(dropout)
  {
    // get scaled qkT
    _bias = nullptr;
    _attention = nullptr;
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

    return _value;
  }

private:
  void init()
  {
    if (_attention) return;

    int S = _q().rows();
    int D = _q().cols();

    // get scaled qkT
    _attention = &(*_graph.new_product(_q, *_graph.new_transpose(_k)) / sqrt(D));

    // attention bias
    _bias = _graph.new_constant(S, S);

    // transpose rows to columns for split/join
    _attention = _graph.new_transpose(*_attention + *_bias);

    // apply softmax on qkT rows transposed to columns
    Function* softmax = nullptr;
    for (int r=0; r<S; r++)
    {
      auto softmax_row = _graph.new_softmax(*_graph.new_split(*_attention, 0,r, D,1));
      if (softmax)
      {
        softmax = _graph.new_join(*softmax, *softmax_row, D,r+1);
      }
      else
      {
        softmax = softmax_row;
      }
    }

    // transpose joined softmax columns back to rows
    _attention = _graph.new_transpose(*softmax);

    if (_dropout > 0)
    {
      _attention = _graph.new_dropout(*_attention, _dropout);
    }

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
