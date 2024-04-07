/*
 * Copyright (c) 2024 Greg Padiasek
 * Distributed under the terms of the MIT License.
 * See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT
 */

#include "main/graph.hh"
#include "main/optimizer.hh"

using namespace seegnify;

static const int SIZE = 5;

int main(int argc, char* argv[]) {

  // create graph and optimizer
  Graph g;
  auto& x = *g.new_constant(1, SIZE);
  auto& y = *g.new_linear(x, SIZE, SIZE);
  Adam opt(g.variables(), 0.001);

  // create loss function
  auto& yhat = *g.new_constant(1, SIZE);
  auto& diff = *g.new_sub(y, yhat);
  auto& loss = *g.new_sum(diff * diff);

  // set target variables
  Tensor W = Tensor::Random(SIZE,SIZE);
  Tensor b = Tensor::Random(1, SIZE);

  // set training params
  int epochs = 100;
  int batch = 1000;
  
  // run training loop
  for (int e=0; e<epochs; e++)
  {
    float error = 0;
    for (int i=0; i<batch; i++)
    {
      // reset input
      g.recache();

      // create radnom regression sample
      Tensor tx = Tensor::Random(1, SIZE);
      yhat.value() = tx * W.transpose() + b;
      x.value() = tx;

      // collect loss
      Tensor l = loss.forward();
      error += l(0);

      // update gradients
      g.backward(loss, Tensor::Ones(1,1));

      // update weights
      opt.update();

      // clear gradients
      g.zero_grad();
    }
        
    // report progress
    std::cout << "error: " << error / batch << std::endl;
    if (error / batch < 1e-5)
    {
      std::cout << "training complete" <<  std::endl;    
      break;
    }    
  }

  /*  
  std::cout << "target variables:" << std::endl;
  for (const auto& t: {W, b})
  {
    std::cout << t.rows() << "x" << t.cols() << std::endl;
    std::cout << t << std::endl;
  }
    
  std::cout << "learned variables:" << std::endl;
  for (auto v: g.variables())
  {
    auto& t = v->value();
    std::cout << t.rows() << "x" << t.cols() << std::endl;
    std::cout << t << std::endl;
  }
  */
  
  return 0;
}
