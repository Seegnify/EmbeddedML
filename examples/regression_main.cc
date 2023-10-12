#include "regression.hh"

#include "main/graph.hh"
#include "main/optimizer.hh"

using namespace seegnify;

int main(int argc, char* argv[]) {
  Graph g;
  auto& rng = g.random();

  float a = 2 * g.random().uniform_dec(-1, 1);
  float b = 3 * g.random().uniform_dec(-1, 1);

  //a = 0;
  //b = 300;

  std::cout << "input a=" << a << std::endl;
  std::cout << "input b=" << b << std::endl;

  std::vector<float> x_coord;
  std::vector<float> y_coord;

  int N = 100;
  for (int x=0; x<N; x++)
  {
    float r = g.random().uniform_dec(-1, 1);
    float y = a * x + b + r;
    x_coord.push_back(x);
    y_coord.push_back(y);
  }

  a = 0.0;
  b = 0.0; 

  int ret = linear_regression(x_coord, y_coord, a, b, 5000, 0.0001);

  std::cout << "output a=" << a << std::endl;
  std::cout << "output b=" << b << std::endl;

  return ret;
}