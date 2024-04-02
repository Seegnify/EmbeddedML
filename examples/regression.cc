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

#include "main/optimizer.hh"
#include "utils/training.hh"
#include "regression.hh"

namespace seegnify {

///////////////////////////////////
// training instance implementation
///////////////////////////////////
class RegressionClient : public Training
{
public:
  RegressionClient(int worker) : Training(worker)
  {
    std::cout << "Regression test worker " << worker << std::endl;

    // get graph instance
    Graph& g = graph();

    // define model
    _model = new RegressionModel(g);

    // create loss function
    _yhat = g.new_constant(1, SIZE);
    auto& diff = *g.new_sub(_model->output(), *_yhat);
    _loss = g.new_mean(diff * diff);
  
    // create optimizer
    _optimizer = new Adam(g.variables(), 0.001);

    // initialize variables for regression samples
    _W = Tensor::Random(SIZE, SIZE) / (SIZE * SIZE);
    _b = Tensor::Random(1, SIZE) / SIZE;
  }

  ~RegressionClient()
  {
    delete _model;
    delete _optimizer;
  }

  virtual void batch_train()
  {
    // restore references
    auto& g = graph();
    auto& x = _model->input();
    auto& y = _model->output();
    auto& yhat = *_yhat;
    auto& loss = *_loss;
    auto& opt = *_optimizer;

    int batch = 1000;
    float error = 0;
    for (int i=0; i<batch; i++)
    {
      // reset input
      g.recache();

      // create radnom regression sample
      Tensor tx = Tensor::Random(1, SIZE);
      yhat.value() = tx * _W.transpose() + _b;
      x.value() = tx;

      // collect loss
      Tensor l = loss.forward();
      error += l(0);

      // update gradients
      g.backward(loss, 0.01 * Tensor::Ones(1,1));

      // update weights
      opt.update();

      // clear gradients
      g.zero_grad();
    }

    // report progress
    if (worker() == 0)
    {
      std::cout << "error: " << error / batch << std::endl;
      if (error / batch < 1e-5)
      {
        std::cout << "training complete" <<  std::endl;
        exit(0);      
      }      
    }    
  }

private:
  RegressionModel *_model;
  Optimizer *_optimizer;
  Constant  *_yhat;
  Function  *_loss;
  Tensor _W;
  Tensor _b;
};

///////////////////////////////////
extern "C" { // export training API
///////////////////////////////////

// create instance for 
DLL_EXPORT Training* create(int idx)
{
  return new RegressionClient(idx);
}

DLL_EXPORT void destroy(Training* ptr)
{
  delete (RegressionClient*)ptr;
}

///////////////////////////////////
} // export training API
///////////////////////////////////

} /* namespace */


