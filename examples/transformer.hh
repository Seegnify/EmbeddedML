/*
 * Copyright 2020-2023 Greg Padiasek. All Rights Reserved.
 */

#ifndef _TRANSFORMER_MODEL_H_
#define _TRANSFORMER_MODEL_H_

#include "main/graph.hh"
#include "main/storage.hh"


using namespace seegnify;


///////////////////////////////////
// Attention
///////////////////////////////////
class Attention : public Function
{
public:
  Attention(Graph& g, Function& q, Function& k, Function& v) :
  Function(g), _k(k)
  {
    _k_dim = g.new_constant(1,1);

    auto& qkT = *g.new_product(q, *g.new_transpose(_k));

    auto& k_dim = *g.new_power(*g.new_broadcast(*_k_dim, qkT), -0.5);

    auto& softmax = *g.new_softmax(*g.new_mul(qkT, k_dim));

    _attention = g.new_product(softmax, v);

    _attention->derivative(_graph.new_iderivative(*this));
  }

  virtual const Tensor& forward()
  {
    if (_value.size() > 0) return _value;

    _k_dim->value() << _k.forward().cols();

    _value = _attention->forward();

    return _value;
  }
  
protected:
  Function& _k;
  Constant* _k_dim;
  Function* _attention;
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
