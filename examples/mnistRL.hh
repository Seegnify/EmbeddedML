/*
 * Copyright (c) 2024 Greg Padiasek
 * Distributed under the terms of the MIT License.
 * See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT
 */

#ifndef _SEEGNIFY_MNISTRLMODEL_H_
#define _SEEGNIFY_MNISTRLMODEL_H_

#include "main/graph.hh"

// MNIST data
#define DATA_ROWS   28
#define DATA_COLS   28
#define LABELS      10

// actions
#define NUM_OF_ACTIONS 8

// RL model params
#define STEPS   10
#define HIDDEN  500
#define INPUT   (3 * VIEW_ROWS * VIEW_COLS)
#define OUTPUT  (LABELS + NUM_OF_ACTIONS)

// RL env params
#define VIEW_ROWS   16
#define VIEW_COLS   16

using namespace seegnify;

class MNISTRLModel
{
public:
  MNISTRLModel(Graph& g)
  {
    // graph dimensions
    const int SIZE = INPUT + HIDDEN + OUTPUT;

    // network hidden input
    Function *h1 = g.new_constant(SIZE, 1);
    Function *h2 = g.new_constant(SIZE, 1);
    Function *h3 = g.new_constant(SIZE, 1);
    h1->value() = Tensor::Zero(SIZE, 1);
    h2->value() = Tensor::Zero(SIZE, 1);
    h3->value() = Tensor::Zero(SIZE, 1);

    // network hidden state
    Function *c1(h1), *c2(h2), *c3(h3);

    // recurrent network over time
    for (int t=0; t<STEPS; t++)
    {
      // input
      _input.push_back(g.new_constant(INPUT, 1));
      auto x = _input.back();

      h1 = g.new_tanh(*g.new_linear(*x, INPUT, SIZE));

      h2 = g.new_gru(*h1, *h2, SIZE, SIZE);

      h3  = g.new_gru(*h2, *h3, SIZE, SIZE);

      auto y = g.new_linear(*h3, SIZE, OUTPUT);

      // action policy
      _policy.push_back(g.new_softmax(*y));
    }
  }

  Constant& input(int t) { return *_input[t]; }

  Function& policy(int t) { return *_policy[t]; }

private:
  // graph references
  std::vector<Constant*> _input;
  std::vector<Function*> _policy;
};

#endif /* _SEEGNIFY_MNISTRLMODEL_H_ */
