/*
 * Copyright (c) 2024 Greg Padiasek
 * Distributed under the terms of the MIT License.
 * See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT
 */

#ifndef _SEEGNIFY_MNISTMODEL_H_
#define _SEEGNIFY_MNISTMODEL_H_

#include "main/graph.hh"

// meta parameters
#define INPUT  (28 * 28)
#define OUTPUT 10
#define HIDDEN 100

using namespace seegnify;

class MNISTModel
{
public:
  MNISTModel(Graph& g)
  {
    // input
    _x = g.new_constant(INPUT, 1);
    _x->value() = Tensor::Zero(INPUT, 1);

    // network
    auto _y_logits = g.new_linear(*_x, INPUT, OUTPUT);
    _y = g.new_softmax(*_y_logits);
  }

  Constant& input() { return *_x; }

  Function& output() { return *_y; }

  Function& output_logits() { return *_y_logits; }

private:
  Constant *_x;
  Function *_y;
  Function *_y_logits;
};

#endif /* _SEEGNIFY_MNISTMODEL_H_ */
