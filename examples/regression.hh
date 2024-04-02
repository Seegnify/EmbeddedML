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

#ifndef _SEEGNIFY_REGRESSION_
#define _SEEGNIFY_REGRESSION_

#include "main/graph.hh"

namespace seegnify {

// meta parameters
#define SIZE 5

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
