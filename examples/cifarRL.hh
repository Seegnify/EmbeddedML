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

#ifndef _SEEGNIFY_CIFARRLMODEL_H_
#define _SEEGNIFY_CIFARRLMODEL_H_

#include "main/graph.hh"

// CIFAR data
#define DATA_ROWS   32
#define DATA_COLS   32
#define LABELS      10

// actions
#define NUM_OF_ACTIONS 8

// RL env params
#define VIEW_ROWS   16
#define VIEW_COLS   16

// RL mpdel params
#define DEPTH       2
#define STEPS       10
#define HIDDEN      50
#define INPUT       (3 * VIEW_ROWS * VIEW_COLS)
#define OUTPUT      (LABELS + NUM_OF_ACTIONS)

// termination codintions
#define MAX_ACTIONS 1000
#define MIN_REWARD -1000

using namespace seegnify;

class CIFARRLModel
{
public:
  CIFARRLModel(Graph& g)
  {
    // model dimensions
    const int SIZE = INPUT + HIDDEN + OUTPUT;

    // network hidden input
    Function *hidden[DEPTH];
    Function *cell[DEPTH];
    for (int i=0; i<DEPTH; i++)
    {
      hidden[i] = g.new_constant(SIZE, 1);
      hidden[i]->value() = Tensor::Zero(SIZE, 1);
      cell[i] = hidden[i];
    }

    // recurrent network in time
    for (int t=0; t<STEPS; t++)
    {
      // time step input
      _input.push_back(g.new_constant(INPUT, 1));
      auto x = _input.back();

      auto l0 = g.new_linear(*x, INPUT, SIZE, "INPUT");
      auto h0 = g.new_tanh(*l0);

      for (int i=0; i<0; i++)
      {
        char name[32];
        sprintf(name, "INPUT-%d", i+1);
        l0 = g.new_linear(*h0, SIZE, SIZE, name);
        h0 = g.new_tanh(*l0);
      }
      
      Function *h_x = h0;
      for (int i=0; i<DEPTH; i++)
      {
        char name[32];
        sprintf(name, "GU-%d", i+1);
        auto h = g.new_lstm(*h_x, *hidden[i], *cell[i], SIZE, SIZE);
        g.name(h, name);
        hidden[i] = h;
        cell[i] = &h->cell();
        h_x = h;
      }

      auto y = g.new_linear(*h_x, SIZE, OUTPUT, "ACTION");

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

#endif /* _SEEGNIFY_CIFARRLMODEL_H_ */
