/*
 * Copyright (c) 2024 Greg Padiasek
 * Distributed under the terms of the MIT License.
 * See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT
 */

#ifndef _SEEGNIFY_REGRESSION_
#define _SEEGNIFY_REGRESSION_

#include "main/graph.hh"

namespace seegnify {

// meta parameters
#define SIZE 200

/////////////////////////////////////////////
// model definition and graph referece holder
/////////////////////////////////////////////
class RegressionModel
{
public:
  RegressionModel(Graph& g)
  {    
    // input x
    _x = g.new_constant(1, SIZE);

    // ouput y = x * W.T + b
    _y = g.new_linear(*_x, SIZE, SIZE);
  }

  Constant& input() { return *_x; }

  Linear& output() { return *_y; }

protected:
  Constant *_x;
  Linear *_y;
};

} /* namespace */

#endif /* _SEEGNIFY_REGRESSION_ */
