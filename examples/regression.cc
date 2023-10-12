#include "regression.hh"

#include "main/graph.hh"
#include "main/optimizer.hh"

using namespace seegnify;

int linear_regression(
  const std::vector<float>& x_coords,
  const std::vector<float>& y_coords,
  float& a,
  float& b,
  int max_iter,
  float learining_rate
)
{
  if (x_coords.size() != y_coords.size()) return -1;

  if (x_coords.size() == 0) return -2;

  // size
  int N = 1;
  Graph g;

  // x
  Constant& x = *g.new_constant(N, 1);

  // y = W * x + b
  auto& y = *g.new_linear(x, N, N);
  //y.W().value()(0) = 2.0;
  //y.b().value()(0) = 1.0;

  // target y
  Constant& y_hat = *g.new_constant(N, 1);

  // Loss
  auto& diff = *g.new_sub(y_hat, y);
  auto& pow2 = *g.new_mul(diff, diff);
  auto& loss = *g.new_sum(pow2);
  //auto& abs = *g.new_abs(diff); // does not work for linear regression
  //auto& loss = *g.new_sum(abs);

  SGD opt(g.variables(), learining_rate);

  for (int epoch=0; epoch<max_iter; epoch++)
  {
    auto size = x_coords.size();
    for (auto i=0; i<size; i++)  
    {
      g.recache();

      // create sample
      y_hat.value()(0) = y_coords[i];
      //std::cout << "y_coord=" << y_hat.value() << std::endl;

      // reset input
      x.value()(0) = x_coords[i];
      //std::cout << "x_coord=" << x.value() << std::endl;

      Tensor l = loss.forward();
      //std::cout << "loss=" << l << std::endl;

      // update gradients
      g.backward(loss, Tensor::Ones(1,1));

      // update weights
      opt.update();

      // clear gradients
      g.zero_grad();
    }
  }

  a = y.W().value()(0);
  b = y.b().value()(0);

  return 0;
}
