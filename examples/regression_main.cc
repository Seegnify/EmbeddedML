#include "regression.hh"

#include "main/graph.hh"
#include "main/optimizer.hh"

using namespace seegnify;

int main(int argc, char* argv[]) {
  float a = 2.5;
  float b = -3.1;

  std::cout << "input a=" << a << std::endl;
  std::cout << "input b=" << b << std::endl;

  std::vector<float> x_coord;
  std::vector<float> y_coord;

  int N = 100;
  Graph g;
  auto& rng = g.random();
  for (int x=0; x<N; x++)
  {
    float r = g.random().uniform_dec(-1, 1);
    float y = a * x + b + r;
    y = 300;
    x_coord.push_back(x);
    y_coord.push_back(y);
  }

  a = 0.0;
  b = 0.0; 

  int ret = linear_regression(x_coord, y_coord, a, b, 500, 0.001);

  std::cout << "output a=" << a << std::endl;
  std::cout << "output b=" << b << std::endl;

  return ret;
}