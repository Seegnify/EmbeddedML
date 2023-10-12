#include <vector>

int linear_regression(
  const std::vector<float>& x,
  const std::vector<float>& y,
  float& a,
  float& b,
  int max_iter,
  float learining_rate
);
