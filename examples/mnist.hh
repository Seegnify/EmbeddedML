/*
 * Copyright 2020-2021 Greg Padiasek and Seegnify <http://www.seegnify.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
