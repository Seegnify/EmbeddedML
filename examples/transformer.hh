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
  // q - query vectors
  // k - key vectors
  // v - value vectors
  // S - sequence length
  // D - vector dimension
  Attention(Graph& g, Function& q, Function& k, Function& v, int S, int D) :
  Function(g)
  {
    // get scaled qkT columns
    auto& qkT = *g.new_product(q, *g.new_transpose(k)) / sqrt(D);

    // transpose rows to columns for split/join
    auto& qkT_rows = *g.new_transpose(qkT);

    // apply softmax on qkT rows transposed to columns 
    Function* softmax = nullptr;
    for (int r=0; r<S; r++)
    {
      auto softmax_row = g.new_softmax(*g.new_split(qkT_rows, 0,r, D,1));
      if (softmax)
      {
        softmax = g.new_join(*softmax, *softmax_row, D,r+1);
      }
      else
      {
        softmax = softmax_row;
      }
    }

    // transpose joined columns back to rows
    softmax = g.new_transpose(*softmax);

    _attention = g.new_product(*softmax, v);

    _attention->derivative(_graph.new_iderivative(*this));
  }

  virtual const Tensor& forward()
  {
    if (_value.size() > 0) return _value;

    _value = _attention->forward();

    return _value;
  }
  
protected:
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
