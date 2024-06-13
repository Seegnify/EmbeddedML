/*
 * Copyright (c) 2024 Greg Padiasek
 * Distributed under the terms of the MIT License.
 * See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT
 */

#include <iostream>
#include <iterator>

// Array.erf
#include <unsupported/Eigen/SpecialFunctions>

#include "graph.hh"

namespace seegnify {

///////////////////////////////////////////
// Function operators
///////////////////////////////////////////

Function& scalar(Graph&g, DTYPE y)
{
  auto& s = *g.new_constant(1,1);
  s.value() << y;
  return s;
}

Function& power(Function& x, DTYPE y)
{
  return *x.graph().new_power(x, y);
}

Function& product(Function& x, Function& y)
{
  return *x.graph().new_product(x, y);
}

Function& operator+(Function& x, Function& y)
{
  return *x.graph().new_add(x, y);
}

Function& operator-(Function& x, Function& y)
{
  return *x.graph().new_sub(x, y);
}

Function& operator*(Function& x, Function& y)
{
  return *x.graph().new_mul(x, y);
}

//Function& Function::operator-()
Function& operator-(Function& x)
{
  auto& s = scalar(x.graph(), -1);
  auto& c = *x.graph().new_broadcast(s, x);
  return *x.graph().new_mul(c, x);
}

// DTYPE operators
Function& operator+(DTYPE x, Function& y)
{
  auto& s = scalar(y.graph(), x);
  auto& c = *y.graph().new_broadcast(s, y);
  return *y.graph().new_add(c, y);
}

Function& operator+(Function& x, DTYPE y)
{
  auto& s = scalar(x.graph(), y);
  auto& c = *x.graph().new_broadcast(s, x);
  return *x.graph().new_add(x, c);
}

Function& operator-(DTYPE x, Function& y)
{
  auto& s = scalar(y.graph(), x);
  auto& c = *y.graph().new_broadcast(s, y);
  return *y.graph().new_sub(c, y);
}

Function& operator-(Function& x, DTYPE y)
{
  auto& s = scalar(x.graph(), y);
  auto& c = *x.graph().new_broadcast(s, x);
  return *x.graph().new_sub(x, c);
}

Function& operator*(DTYPE x, Function& y)
{
  auto& s = scalar(y.graph(), x);
  auto& c = *y.graph().new_broadcast(s, y);
  return *y.graph().new_mul(c, y);
}

Function& operator*(Function& x, DTYPE y)
{
  auto& s = scalar(x.graph(), y);
  auto& c = *x.graph().new_broadcast(s, x);
  return *x.graph().new_mul(x, c);
}

Function& operator/(Function& x, DTYPE y)
{
  return x * (1.0/y);
}

Function& operator/(DTYPE x, Function& y)
{
  auto& s = scalar(y.graph(), x);
  auto& c = *y.graph().new_broadcast(s, y);
  return c * power(y, -1);
}

Function& operator/(Function& x, Function& y)
{
  return x * power(y, -1);
}

///////////////////////////////////////////
// Tensor routines
///////////////////////////////////////////

static Tensor ATB(const Tensor& A, const Tensor& B)
{
  if (B.cols() == 1)
    return (ConstRowVectorMap(B.data(), B.size()) * A).transpose();
  else
    return A.transpose() * B;
}

// compute sparse product A.T*B at non-zero elements of a reference tensor R
static SparseTensor ATB(
  const Tensor& A,
  const Tensor& B,
  const SparseTensor& R
)
{
  auto AT = A.transpose();
  auto BT = B.transpose();
  SparseTensor abt(R.rows(), R.cols());
  for (int k=0; k < R.outerSize(); ++k)
  {
      for (SparseTensor::InnerIterator it(R,k); it; ++it)
      {
          auto row = it.row();
          auto col = it.col();
          abt.coeffRef(row, col) = AT(row) * BT(col);
      }
  }
  return abt;
}

static Tensor ABT(const Tensor& A, const Tensor& B)
{
  if (B.cols() == 1)
    return A * ConstRowVectorMap(B.data(), B.size());
  else
    return A * B.transpose();
}

// compute sparse product A*B.T at non-zero elements of a reference tensor R
static SparseTensor ABT(
  const Tensor& A,
  const Tensor& B,
  const SparseTensor& R
)
{
  SparseTensor abt(R.rows(), R.cols());
  for (int k=0; k < R.outerSize(); ++k)
  {
      for (SparseTensor::InnerIterator it(R,k); it; ++it)
      {
          auto row = it.row();
          auto col = it.col();
          abt.coeffRef(row, col) = A(row) * B(col);
      }
  }
  return abt;
}

static Tensor clip(const Tensor& t, DTYPE min_val, DTYPE max_val)
{
  auto lo = Tensor::Constant(t.rows(), t.cols(), min_val);
  auto up = Tensor::Constant(t.rows(), t.cols(), max_val);
  return t.array().min(up.array()).max(lo.array());
}

///////////////////////////////////////////
// Function impl
///////////////////////////////////////////

Function::Function(Graph& graph) : _graph(graph)
{
  _backprop = true;
}

// backward traversal
const Tensor& Function::backward()
{
  // if no value is set, there is no gradient
  if (!_value.size()) throw NoValueException();

  // update and cache gradient
  if (!_gradient.size())
  {
    _gradient = Tensor::Zero(_value.rows(), _value.cols());

    if (_backprop) _graph.aggregate(_gradient, _backward);
  }

  return _gradient;
}

// recache function value and gradient
void Function::recache()
{
  _value.resize(0,0);
  _gradient.resize(0,0);
}

// set function name
void Function::name(const std::string& name, bool unique)
{
  if (unique)
  {
    auto scope_name = _graph.scope_name() + name;

    size_t index = 0;

    auto unique_name = scope_name + "." + std::to_string(index);

    while(_graph.function(unique_name))
    {
      unique_name = scope_name + "." + std::to_string(++index);
    }

    _name = unique_name;
  }
  else
  {
    _name = name;
  }
}

///////////////////////////////////////////
// Identity Derivarive impl
///////////////////////////////////////////

IDerivative::IDerivative(Graph& graph, Function& base) : Function(graph)
{
  iforward(&base);
}

// dFdx = base.dFdx
const Tensor& IDerivative::forward()
{
  // return cached value
  if (_value.size()) return _value;

  auto& g = _forward[0]->backward();

  // update gradient value
  _value = g;

  // return gradient value
  return _value;
}

///////////////////////////////////////////
// Function Rowwise
///////////////////////////////////////////

Rowwise::Rowwise(Graph& graph, Function& x, int rows, int cols,
std::function<Function*(Function& block)> ctor) : Function(graph)
{
  iforward(&x);

  _y = nullptr;

  // split x by row and join rows
  for (int r=0; r<rows; r++)
  {
    auto row = ctor(*graph.new_split(x, r,0, 1,cols));
    if (_y)
    {
      _y = graph.new_join(*_y, *row, r+1,cols); // row major
    }
    else
    {
      _y = row;
    }
  }

  _y->ibackward(graph.new_iderivative(*this));
}

Rowwise::Rowwise(Graph& graph, Function& x, int rows, int cols,
std::function<Function*(Function& block)> shared_ctor,
std::function<Function*(Function& block, Function& shared)> ctor) :
Function(graph)
{
  iforward(&x);

  // row-wise function with shared weights
  Function* shared = nullptr;

  // split x by row and join rows
  for (int r=0; r<rows; r++)
  {
    if (shared)
    {
      auto row = ctor(*graph.new_split(x, r,0, 1,cols), *shared);
      _y = graph.new_join(*_y, *row, r+1,cols); // row major
    }
    else
    {
      shared = shared_ctor(*graph.new_split(x, r,0, 1,cols));
      _y = shared;
    }
  }

  _y->ibackward(graph.new_iderivative(*this));
}


// F = f(x rowwise)
const Tensor& Rowwise::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // update value
  _value = _y->forward();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Colwise
///////////////////////////////////////////

Colwise::Colwise(Graph& graph, Function& x, int rows, int cols,
std::function<Function*(Function& block)> ctor) :
Function(graph)
{
  iforward(&x);

  // transpose and apply row-wise ctor, then transpose again
  _y = graph.new_transpose(
    *graph.new_rowwise(
      *graph.new_transpose(x), rows, cols, ctor
    )
  );

  _y->ibackward(graph.new_iderivative(*this));
}

Colwise::Colwise(Graph& graph, Function& x, int rows, int cols,
std::function<Function*(Function& block)> shared_ctor,
std::function<Function*(Function& block, Function& shared)> ctor) :
Function(graph)
{
  iforward(&x);

  // transpose and apply row-wise ctor, then transpose again
  _y = graph.new_transpose(
    *graph.new_rowwise(
      *graph.new_transpose(x), rows, cols, shared_ctor, ctor
    )
  );

  _y->ibackward(graph.new_iderivative(*this));
}

// F = f(x colwise)
const Tensor& Colwise::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // update value
  _value = _y->forward();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Constant
///////////////////////////////////////////

Constant::Constant(Graph& graph, size_t rows, size_t cols) :
Function(graph)
{
  _value.resize(rows, cols);
}

///////////////////////////////////////////
// Function Variable
///////////////////////////////////////////

// random variable
Variable::Variable(Graph& graph, size_t rows, size_t cols) :
Function(graph)
{
  // Xavier initialization
  float stddev = (cols == 1) ? (1.0 / rows) : (1.0 / cols);
  auto random = [&]() { return graph.random().normal_dec(0, stddev); };
  _value = Tensor::NullaryExpr(rows, cols, random);
}

// accumulate gradient
const Tensor& Variable::backward()
{
  // if no value is set, there is no gradient
  if (!_value.size()) throw NoValueException();

  // initialize gradient to zero
  if (!_gradient.size())
  {
    _gradient = Tensor::Zero(_value.rows(), _value.cols());
  }

  if (_backprop) _graph.aggregate(_gradient, _backward);

  return _gradient;
}

///////////////////////////////////////////
// Function Broadcast
///////////////////////////////////////////

Broadcast::Broadcast(Graph& graph, Function& x, Function& target) :
Function(graph)
{
  iforward(&x);
  iforward(&target);

  // Broadcast Backward function
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Broadcast& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      graph.keep(this);
    }

    // dFdx = 1
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& x = _forward[1]->forward();

      // broadcast row-wise
      if (x.rows() == 1 && x.cols() > 1)
      {
        Tensor dFdx = g.colwise().sum();
        _value = dFdx;
      }
      // broadcast col-wise
      else if (x.rows() > 1 && x.cols() == 1)
      {
        Tensor dFdx = g.rowwise().sum();
        _value = dFdx;
      }
      // broadcast any size
      else
      {
        Tensor dFdx = Tensor::Constant(x.rows(), x.cols(), g.sum());
        _value = dFdx;
      }

      // return gradient value
      return _value;
    }
  };

  x.ibackward(new Derivative_x(_graph, *this));
}

// F = x (x broadcast to target)
const Tensor& Broadcast::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get source and target
  auto& x = _forward[0]->forward();
  auto& t = _forward[1]->forward();
  auto rows = t.rows();
  auto cols = t.cols();

  // broadcast row-wise
  if (x.rows() == 1 && x.cols() > 1)
  {
    _value = Tensor::Zero(rows, cols);
    _value.rowwise() += ConstRowVectorMap(x.data(), x.size());
  }
  // broadcast col-wise
  else if (x.rows() > 1 && x.cols() == 1)
  {
    _value = Tensor::Zero(rows, cols);
    _value.colwise() += ConstColVectorMap(x.data(), x.size());
  }
  // broadcast any size
  else
  {
    _value = Tensor::Constant(rows, cols, x.sum());
  }

  return _value;
}

///////////////////////////////////////////
// Function Reshape
///////////////////////////////////////////

Reshape::Reshape(Graph& graph, Function& x, size_t rows, size_t cols) :
Function(graph), _rows(rows), _cols(cols)
{
  iforward(&x);

  // Reshape Backward function
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Reshape& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      graph.keep(this);
    }

    // dFdx = 1
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& x = _forward[1]->forward();

      // pass the reshaped gradient through (default is ColMajor)
      auto dFdx = g.reshaped<Eigen::RowMajor>(x.rows(), x.cols());

      // update gradient value
      _value = dFdx;

      // return gradient value
      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
}

// F = x
const Tensor& Reshape::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _forward[0]->forward();

  // update value (reshaped() default is ColMajor)
  _value = x.reshaped<Eigen::RowMajor>(_rows, _cols);

  return _value;
}

///////////////////////////////////////////
// Function Split
///////////////////////////////////////////

Split::Split(Graph& graph, Function& x, size_t r, size_t c, size_t rows, size_t cols) :
Function(graph), _r(r), _c(c), _rows(rows), _cols(cols)
{
  iforward(&x);

  // Split Backward function
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Split& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      graph.keep(this);
    }

    // dFdx = 1
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto base = (Split*)_forward[0];

      auto& g = _forward[0]->backward();
      auto& x = _forward[1]->forward();

      auto r = base->_r;
      auto c = base->_c;
      auto rows = base->_rows;
      auto cols = base->_cols;

      // pass the gradient through the block
      Tensor dFdx = Tensor::Zero(x.rows(), x.cols());
      dFdx.block(r, c, rows, cols) = g;

      // update gradient value
      _value = dFdx;

      // return gradient value
      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
}

// F = block(x)
const Tensor& Split::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _forward[0]->forward();

  // update value
  _value = x.block(_r, _c, _rows, _cols);

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Join
///////////////////////////////////////////

Join::Join(Graph& graph, Function& x, Function& y, size_t rows, size_t cols) :
Function(graph), _rows(rows), _cols(cols)
{
  iforward(&x);
  iforward(&y);

  // Join Derivative with respect to X
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Join& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      graph.keep(this);
    }

    // dFdx = 1
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& x = _forward[1]->forward();

      // split g into gx and gy (g data is already flat)
      ConstTensorMap gx(g.data(), x.rows(), x.cols());

      // update gradient value
      _value = gx;
      return _value;
    }
  };

  // Join Derivative with respect to Y
  class Derivative_y : public Function
  {
  public:
    Derivative_y(Graph& graph, Join& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      iforward(base._forward[1]);
      graph.keep(this);
    }

    // dFdy = 1
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& x = _forward[1]->forward();
      auto& y = _forward[2]->forward();

      // split g into gx and gy (g data is already flat)
      ConstTensorMap gy(g.data() + x.size(), y.rows(), y.cols());

      // update gradient value
      _value = gy;
      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
  y.ibackward(new Derivative_y(graph, *this));
}

// F = join(x)
const Tensor& Join::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _forward[0]->forward();
  auto& y = _forward[1]->forward();

  // join x and y into a flat vector
  ConstRowVectorMap vx(x.data(), x.size());
  ConstRowVectorMap vy(y.data(), y.size());
  RowVector xy(x.size() + y.size());
  xy.leftCols(x.size()) = vx;
  xy.rightCols(y.size()) = vy;

  // reshape flat vector into output shape
  _value = TensorMap(xy.data(), _rows, _cols);

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Min
///////////////////////////////////////////

Min::Min(Graph& graph, Function& x, Function& y) : Function(graph)
{
  iforward(&x);
  iforward(&y);

  // Derivative with respect to X
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Min& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      iforward(base._forward[1]);
      graph.keep(this);
    }

    // dFdx = 1 if x < y, 0 otherwise
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& x = _forward[1]->forward();
      auto& y = _forward[2]->forward();

      // compute gradient mask
      auto one = Tensor::Ones(x.rows(), x.cols());
      auto zero = Tensor::Zero(x.rows(), x.cols());
      auto dFdx_mask = (x.array() < y.array()).select(one, zero);
      auto dFdy_mask = (x.array() > y.array()).select(one, zero);

      // update gradient value
      _value = g.array() * dFdx_mask.array();
      return _value;
    }
  };

  // Derivative with respect to Y
  class Derivative_y : public Function
  {
  public:
    Derivative_y(Graph& graph, Min& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      iforward(base._forward[1]);
      graph.keep(this);
    }

    // dFdy = 1 if x > y, 0 otherwise
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& x = _forward[1]->forward();
      auto& y = _forward[2]->forward();

      // compute gradient mask
      auto one = Tensor::Ones(x.rows(), x.cols());
      auto zero = Tensor::Zero(x.rows(), x.cols());
      auto dFdx_mask = (x.array() < y.array()).select(one, zero);
      auto dFdy_mask = (x.array() > y.array()).select(one, zero);

      // update gradient value
      _value = g.array() * dFdy_mask.array();
      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
  y.ibackward(new Derivative_y(graph, *this));
}

// F = min(x,y)
const Tensor& Min::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _forward[0]->forward();
  auto& y = _forward[1]->forward();

  // update value
  _value = x.array().min(y.array());

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Max
///////////////////////////////////////////

Max::Max(Graph& graph, Function& x, Function& y) : Function(graph)
{
  iforward(&x);
  iforward(&y);

  // Derivative with respect to X
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Max& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      iforward(base._forward[1]);
      graph.keep(this);
    }

    // dFdx = 1 if x > y, 0 otherwise
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& x = _forward[1]->forward();
      auto& y = _forward[2]->forward();

      // compute gradient mask
      auto one = Tensor::Ones(x.rows(), x.cols());
      auto zero = Tensor::Zero(x.rows(), x.cols());
      auto dFdx_mask = (x.array() > y.array()).select(one, zero);

      // update gradient value
      _value = g.array() * dFdx_mask.array();
      return _value;
    }
  };

  // Derivative with respect to Y
  class Derivative_y : public Function
  {
  public:
    Derivative_y(Graph& graph, Max& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      iforward(base._forward[1]);
      graph.keep(this);
    }

    // dFdy = 1 if x < y, 0 otherwise
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& x = _forward[1]->forward();
      auto& y = _forward[2]->forward();

      // compute gradient mask
      auto one = Tensor::Ones(x.rows(), x.cols());
      auto zero = Tensor::Zero(x.rows(), x.cols());
      auto dFdy_mask = (x.array() < y.array()).select(one, zero);

      // update gradient value
      _value = g.array() * dFdy_mask.array();
      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
  y.ibackward(new Derivative_y(graph, *this));
}

// F = max(x,y)
const Tensor& Max::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _forward[0]->forward();
  auto& y = _forward[1]->forward();

  // update value
  _value = x.array().max(y.array());

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Linear
///////////////////////////////////////////

// x dim = [samples x in]
// W dim = [out x in]
// b dim = [samples x out]
Linear::Linear(Graph& graph, Function& x, int in, int out, bool bias) :
Function(graph)
{
  // construct new variables
  _W = graph.new_variable(out, in, "Linear.W");

  // col major
  // _b = (bias) ? graph.new_variable(out, 1) : nullptr;

  // row major
  _b = (bias) ? graph.new_variable(1, out, "Linear.b") : nullptr;

  init(x);
}

Linear::Linear(Graph& graph, Function& x, const Linear& other) :
Function(graph)
{
  // share variables with the "other"
  _W = other._W;
  _b = other._b;

  init(x);
}

void Linear::init(Function& x)
{
  iforward(&x);

  // col major
  // F = W * x + b
  // _y = _graph.new_product(*_W, x);

  // row major
  // F = x * W.T + b
  _y = _graph.new_product(x, *_graph.new_transpose(*_W));

  if (_b)
  {
    _y = _graph.new_add(*_y, *_graph.new_broadcast(*_b, *_y));
  }

  _y->ibackward(_graph.new_iderivative(*this));
}

const Tensor& Linear::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // create cached value
  _value.noalias() = _y->forward();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Product
///////////////////////////////////////////

Product::Product(Graph& graph, Function& x, Function& y) :
Function(graph)
{
  iforward(&x);
  iforward(&y);

  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Product& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[1]);
      graph.keep(this);
    }

    // dFdx = y.T
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& y = _forward[1]->forward();

      auto& dFdx = y;

      // update gradient value
      _value = ABT(g, dFdx);
      return _value;
    }
  };

  // Derivative with respect to y
  class Derivative_y : public Function
  {
  public:
    Derivative_y(Graph& graph, Product& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      graph.keep(this);
    }

    // dFdy = x.T
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& x = _forward[1]->forward();

      auto& dFdy = x;

      // update gradient value
      _value = ATB(dFdy, g);
      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
  y.ibackward(new Derivative_y(graph, *this));
}

// F = x * y
const Tensor& Product::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _forward[0]->forward();
  auto& y = _forward[1]->forward();

  // create cached value
  _value.noalias() = x * y;

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Add
///////////////////////////////////////////

Add::Add(Graph& graph, Function& x, Function& y) : Function(graph)
{
  iforward(&x);
  iforward(&y);

  // Derivative with respect to x or y
  class Derivative_any : public Function
  {
  public:
    Derivative_any(Graph& graph, Add& base) : Function(graph)
    {
      iforward(&base);
      graph.keep(this);
    }

    // dFdx = 1
    // dFdy = 1
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();

      // update gradient value
      _value = g;
      return _value;
    }
  };

  x.ibackward(new Derivative_any(graph, *this));
  y.ibackward(new Derivative_any(graph, *this));
}

// F = x + y
const Tensor& Add::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get inputs
  auto& x = _forward[0]->forward();
  auto& y = _forward[1]->forward();

  // update value
  _value.noalias() = x + y;

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Sub
///////////////////////////////////////////

Sub::Sub(Graph& graph, Function& x, Function& y) : Function(graph)
{
  iforward(&x);
  iforward(&y);

  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Sub& base) : Function(graph)
    {
      iforward(&base);
      graph.keep(this);
    }

    // dFdx = 1
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();

      // update gradient value
      _value = g;
      return _value;
    }
  };

  // Derivative with respect to y
  class Derivative_y : public Function
  {
  public:
    Derivative_y(Graph& graph, Sub& base) : Function(graph)
    {
      iforward(&base);
      graph.keep(this);
    }

    // dFdy = -1
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();

      // update gradient value
      _value = -g;
      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
  y.ibackward(new Derivative_y(graph, *this));
}

// F = x - y
const Tensor& Sub::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get inputs
  auto& x = _forward[0]->forward();
  auto& y = _forward[1]->forward();

  // update value
  _value.noalias() = x - y;

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Mul
///////////////////////////////////////////

Mul::Mul(Graph& graph, Function& x, Function& y) : Function(graph)
{
  iforward(&x);
  iforward(&y);

  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Mul& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[1]);
      graph.keep(this);
    }

    // dFdx = y
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& y = _forward[1]->forward();

      auto& dFdx = y;

      // update gradient value
      _value = g.array() * dFdx.array();

      return _value;
    }
  };

  // Derivative with respect to y
  class Derivative_y : public Function
  {
  public:
    Derivative_y(Graph& graph, Mul& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      graph.keep(this);
    }

    // dFdy = x
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& x = _forward[1]->forward();

      auto& dFdy = x;

      // update gradient value
      _value = g.array() * dFdy.array();

      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
  y.ibackward(new Derivative_y(graph, *this));
}

// F = x * y
const Tensor& Mul::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get inputs
  auto& x = _forward[0]->forward();
  auto& y = _forward[1]->forward();

  // update value
  _value = x.array() * y.array();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Power
///////////////////////////////////////////

Power::Power(Graph& graph, Function& x, Function& y) : Function(graph)
{
  iforward(&x);
  iforward(&y);

  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Power& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      iforward(base._forward[1]);
      graph.keep(this);
    }

    // dFdx = y * pow(F, y-1)
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& F = _forward[0]->forward();
      auto& g = _forward[0]->backward();
      auto& x = _forward[1]->forward();
      auto& y = _forward[2]->forward();

      auto dFdx = y.array() * x.array().pow(y.array() - 1);

      // update gradient value
      _value = g.array() * dFdx.array();
      return _value;
    }
  };

  // Derivative with respect to y
  class Derivative_y : public Function
  {
  public:
    Derivative_y(Graph& graph, Power& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      iforward(base._forward[1]);
      graph.keep(this);
    }

    // dFdy = F * log(x)
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& F = _forward[0]->forward();
      auto& g = _forward[0]->backward();
      auto& x = _forward[1]->forward();
      auto& y = _forward[2]->forward();

      auto dFdy = F.array() * x.array().log();

      // update gradient value
      _value = g.array() * dFdy.array();
      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
  y.ibackward(new Derivative_y(graph, *this));
}

// F = pow(x, y)
const Tensor& Power::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get inputs
  auto& x = _forward[0]->forward();
  auto& y = _forward[1]->forward();

  // update value
  _value = x.array().pow(y.array());

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Tanh
///////////////////////////////////////////

Tanh::Tanh(Graph& graph, Function& x) : Function(graph)
{
  iforward(&x);

  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Tanh& base) : Function(graph)
    {
      iforward(&base);
      graph.keep(this);
    }

    // dFdx = 1 - (tanh(x))^2
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& F = _forward[0]->forward();

      auto dFdx = 1 - F.array() * F.array();

      // update gradient value
      _value = g.array() * dFdx.array();

      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
}

// F = (exp(x) - exp(-x) / (exp(x) + exp(-x))
const Tensor& Tanh::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _forward[0]->forward();

  // update value
  _value = x.array().tanh();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Sigmoid
///////////////////////////////////////////

Sigmoid::Sigmoid(Graph& graph, Function& x) :
Function(graph)
{
  iforward(&x);

  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Sigmoid& base) : Function(graph)
    {
      iforward(&base);
      graph.keep(this);
    }

    // dFdx = F(1 - F)
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& F = _forward[0]->forward();

      auto dFdx = F.array() * (1 - F.array());

      // update gradient value
      _value = g.array() * dFdx.array();

      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
}

// F = 1 / (1 + exp(-x))
const Tensor& Sigmoid::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _forward[0]->forward();

  // update value
  //_value = 1 / (1 + (-x).array().exp());
  // for numerical stability compute sigmoid with tanh
  _value = 0.5 * ((0.5 * x).array().tanh() + 1.0);

  // return value
  return _value;
}

///////////////////////////////////////////
// Function ReLU
///////////////////////////////////////////

ReLU::ReLU(Graph& graph, Function& x) :
Function(graph)
{
  iforward(&x);

  auto& zero = scalar(graph, 0);

  // create relu from function max and constant '0'
  _relu = graph.new_max(x, *graph.new_broadcast(zero, x));

  // let output handle the gradient directly
  _relu->ibackward(_graph.new_iderivative(*this));
}

// F = max(x, 0)
const Tensor& ReLU::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // update value
  _value = _relu->forward();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Dropout
///////////////////////////////////////////

Dropout::Dropout(Graph& graph, Function& x, DTYPE rate) :
Function(graph), _rate(rate), _enabled(true)
{
  iforward(&x);

  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Dropout& base) : Function(graph)
    {
      iforward(&base);
      graph.keep(this);
    }

    // dFdy = x * mask
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto base = (Dropout*)_forward[0];

      auto& g = base->backward();
      auto& m = base->_mask;

      auto dFdx = g.array() * m.array();

      // update gradient value
      _value = dFdx;
      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
}

// F = x * mask
const Tensor& Dropout::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get values
  auto& x = _forward[0]->forward();

  // compute dropout mask
  if (_enabled)
  {
    auto random = [&]() { return _graph.random().uniform_dec(0, 1); };
    auto mask = Tensor::NullaryExpr(x.rows(), x.cols(), random);
    auto zero = Tensor::Zero(x.rows(), x.cols());
    auto ones = Tensor::Ones(x.rows(), x.cols());
    _mask = (mask.array() < _rate).select(zero, ones);
  }
  else
  {
    _mask = Tensor::Ones(x.rows(), x.cols());
  }

  // update value
  _value = x.array() * _mask.array();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Error
///////////////////////////////////////////

Erf::Erf(Graph& graph, Function& x) :
Function(graph)
{
  iforward(&x);

  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Erf& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      graph.keep(this);
    }

    // dFdx = (2 / sqrt(pi)) * exp(-x^2)
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& x = _forward[1]->forward();

      // M_2_SQRTPI = 2/sqrt(pi)
      auto dFdx = M_2_SQRTPI * (-x.array() * x.array()).exp();

      // update gradient value
      _value = dFdx.array() * g.array();

     return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
}

// F = erf(x)
const Tensor& Erf::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _forward[0]->forward();

  // update value
  _value = x.array().erf();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function GeLU
///////////////////////////////////////////

#ifdef APPROXIMATE_GELU

GeLU::GeLU(Graph& graph, Function& x) :
Function(graph)
{
  iforward(&x);

  // M_2_PI = 2 / pi
  //_gelu = &(0.5 * x * (1.0 + *graph.new_tanh(
  //  M_2_PI * (x + 0.044715 * *graph.new_power(x, 3))
  //)));

  _gelu = &(x * *graph.new_sigmoid(1.702 * x));

  _gelu->ibackward(graph.new_iderivative(*this));
}

// F = 0.5 (1 + tanh(√2/π(x + 0.044715 x^3)))
const Tensor& GeLU::forward()
{
  // return cached value
  if (_value.size()) return _value;

  _value = _gelu->forward();

  // return value
  return _value;
}

#else

GeLU::GeLU(Graph& graph, Function& x) :
Function(graph)
{
  iforward(&x);

  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, GeLU& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      graph.keep(this);
    }

    // F = 0.5 * x * (1 + erf(x / sqrt(2)))
    // z(x) = x / sqrt(2)
    // F = 0.5 * x * (1 + erf(z))
    // dFdx = 0.5 * (1 + erf(z)) + 0.5 * x * derfdz / sqrt(2)
    // derfdz = erf(-z^2) * 2 / sqrt(pi)
    // dFdx = 0.5 * (1 + erf(z)) + 0.5 * x * erf(-z^2) * 2 / sqrt(pi) / sqrt(2)
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& x = _forward[1]->forward();

      auto z = x.array() * M_SQRT1_2;
      auto derfdz = M_2_SQRTPI * (-z.array() * z.array()).exp();
      Tensor dFdx = 0.5 * (1 + z.erf()) + 0.5 * x.array() * derfdz * M_SQRT1_2;

      // update gradient value
      _value = g.array() * dFdx.array();

      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
}

// F = 0.5 * x * (1 + erf(x / sqrt(2)))
const Tensor& GeLU::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _forward[0]->forward();

  // update value
  _value = 0.5 * x.array() * (1 + (x.array() * M_SQRT1_2).erf());

  // return value
  return _value;
}

#endif

///////////////////////////////////////////
// Function Softmax
///////////////////////////////////////////

// Derivative approches 0 when error is evently distributed.
// Use LogSoftmax for training and Softmax for inference.
Softmax::Softmax(Graph& graph, Function& x) :
Function(graph)
{
  iforward(&x);

  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Softmax& base) : Function(graph)
    {
      iforward(&base);
      graph.keep(this);
    }

    // dFdx(j,i) = F(i)(K(i,j) - F(j))
    // K(i,j) = 1 for i==j, 0 for i!=j (Kronecker delta)
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& y = _forward[0]->forward();

      // reshape input to flat row vector
      auto F = y.reshaped(1, y.size());

      // use Identity matrix to construct diagonal matrix
      auto I = Tensor::Identity(F.size(), F.size());

      // col major
      //auto dFdx = I * F.asDiagonal() - F * F.transpose();
      //auto gdFdx = (g.asDiagonal() * dFdx).colwise().sum().transpose();

      // row major
      auto dFdx = I * F.asDiagonal() - F.transpose() * F;
      auto gdFdx = (g.asDiagonal() * dFdx).colwise().sum();

      // reshape to input shape
      _value = gdFdx.reshaped(y.rows(), y.cols());

      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
}

// F = exp(x) / sum(exp(x))
const Tensor& Softmax::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input vector
  auto& x = _forward[0]->forward();

  // for numerical stability subtract max(_x) before exponential
  auto exp_x = (x.array() - x.maxCoeff()).exp();

  // create cached value
  _value = exp_x / exp_x.sum();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Softplus
///////////////////////////////////////////

Softplus::Softplus(Graph& graph, Function& x) :
Function(graph)
{
  iforward(&x);

  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Softplus& base) : Function(graph)
    {
      iforward(base._forward[0]);
      graph.keep(this);
    }

    // dFdx = sigmoid(x) = 1 / (1 + exp(-x))
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      // get input
      auto& x = _forward[0]->forward();

      // update gradient
      //_value = 1 / (1 + (-x).array().exp());
      // for numerical stability compute sigmoid with tanh
      _value = 0.5 * (x.array() * 0.5).tanh() + 0.5;

      // return value
      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
}

// F = log(1 + exp(x))
const Tensor& Softplus::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input vector
  auto& x = _forward[0]->forward();

  // _value = log(1+exp(x))
  // log(1+exp(x)) = log(1+exp(x)) - log(exp(x)) + x = log(1+exp(-x)) + x
  // so for numerical stability use: log(1+exp(-abs(x))) + max(x,0)

  // create cached value
  auto zero = Tensor::Zero(x.rows(), x.cols());
  auto max_x_0 = x.array().max(zero.array());
  _value = (1 + (-x.array().abs()).exp()).log() + max_x_0;

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Log-Softmax
///////////////////////////////////////////

LogSoftmax::LogSoftmax(Graph& graph, Function& x) :
Function(graph)
{
  iforward(&x);

  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, LogSoftmax& base) : Function(graph)
    {
      iforward(&base);
      graph.keep(this);
    }

    // dFdx(j,i) = K(i,j) - F(j)
    // K(i,j) = 1 for i==j, 0 for i!=j (Kronecker delta)
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      auto base = (LogSoftmax*)_forward[0];

      auto& g = base->backward();
      auto s = base->softmax();

      // reshape input to flat row vector
      auto S = s.reshaped(1, s.size());

      // use Identity matrix to construct diagonal matrix
      auto I = Tensor::Identity(S.size(), S.size());

      // col-major
      // auto dFdx = I - Tensor::Ones(S.size(), 1) * S.transpose();
      // auto gdFdx = (g.asDiagonal() * dFdx).colwise().sum().transpose();

      // row-major
      auto dFdx = I - Tensor::Ones(S.size(), 1) * S;
      auto gdFdx = (g.asDiagonal() * dFdx).colwise().sum();

      // reshape to input shape
      _value = gdFdx.reshaped(s.rows(), s.cols());

      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
}

// F = exp(x) / sum(exp(x))
const Tensor LogSoftmax::softmax()
{
  // get input vector
  auto& x = _forward[0]->forward();

  // for numerical stability subtract max(_x) before exponential
  auto exp_x = (x.array() - x.maxCoeff()).exp();

  // calculate softmax
  return exp_x / exp_x.sum();
}

// F = log(exp(x) / sum(exp(x))) = x - log(sum(exp(x)))
const Tensor& LogSoftmax::forward()
{
  // return cached value
  if (_value.size()) return _value;

  auto& x = _forward[0]->forward();

  // for numerical stability subtract max(_x) before exponential
  auto x_C = x.array() - x.maxCoeff();

  // create cached value
  _value = x_C - std::log(x_C.exp().sum());

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Log
///////////////////////////////////////////

Log::Log(Graph& graph, Function& x) :
Function(graph)
{
  iforward(&x);

  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Log& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      graph.keep(this);
    }

    // dFdx = 1 / x
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& x = _forward[1]->forward();

      auto dFdx = 1 / x.array();

      // update gradient value
      _value = g.array() * dFdx.array();

      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
}

// F = log(x)
const Tensor& Log::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _forward[0]->forward();

  // update value
  _value = x.array().log();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Abs
///////////////////////////////////////////

Abs::Abs(Graph& graph, Function& x) :
Function(graph)
{
  iforward(&x);

  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Abs& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      graph.keep(this);
    }

    // dFdx = abs(x) / x
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& x = _forward[1]->forward();

      // Taking sing() of input, gives the derivative values,
      // and for x == 0 the derivative becomes 0, which is half
      // of the derivative maximum and minimum value.
      auto dFdx = x.array().sign();

      // update gradient value
      _value = g.array() * dFdx.array();

      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
}

// F = abs(x)
const Tensor& Abs::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _forward[0]->forward();

  // update value
  _value = x.array().abs();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Transpose
///////////////////////////////////////////

Transpose::Transpose(Graph& graph, Function& x) :
Function(graph)
{
  iforward(&x);

  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Transpose& base) : Function(graph)
    {
      iforward(&base);
      graph.keep(this);
    }

    // dFdx = base.dFdx.T
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();

      // update gradient value
      _value = g.transpose();

      // return gradient value
      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
}

// F = x.T
const Tensor& Transpose::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _forward[0]->forward();

  // update value
  _value = x.transpose();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Sum
///////////////////////////////////////////

Sum::Sum(Graph& graph, Function& x) :
Function(graph)
{
  iforward(&x);

  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Sum& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      graph.keep(this);
    }

    // dFdx = 1
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& x = _forward[1]->forward();

      auto dFdx = g(0,0);

      // update gradient value
      _value = Tensor::Constant(x.rows(), x.cols(), dFdx);

     return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
}

// F = sum(x)
const Tensor& Sum::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _forward[0]->forward();

  // update value
  _value.resize(1,1);
  _value(0,0) = x.sum();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Mean
///////////////////////////////////////////

Mean::Mean(Graph& graph, Function& x) :
Function(graph)
{
  iforward(&x);

  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Mean& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      graph.keep(this);
    }

    // dFdx = 1 / N
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& x = _forward[1]->forward();

      auto dFdx = g(0,0) / x.size();

      // update gradient value
      _value = Tensor::Constant(x.rows(), x.cols(), dFdx);

      return _value;
    }
  };

  x.ibackward(new Derivative_x(graph, *this));
}

// F = sum(x) / N
const Tensor& Mean::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _forward[0]->forward();

  // update value
  _value.resize(1,1);
  _value(0,0) = x.sum() / x.size();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function GRU
///////////////////////////////////////////

GRU::GRU(Graph& graph, Function& x, Function& h, int in, int out) :
Function(graph)
{
  // construct new variables (row-major)
  _Wz = graph.new_variable(in, out, "Wz");
  _Uz = graph.new_variable(out, out, "Uz");
  _bz = graph.new_variable(1, out, "bz");

  _Wr = graph.new_variable(in, out, "Wr");
  _Ur = graph.new_variable(out, out, "Ur");
  _br = graph.new_variable(1, out, "br");

  _Wh = graph.new_variable(in, out, "Wh");
  _Uh = graph.new_variable(out, out, "Uh");
  _bh = graph.new_variable(1, out, "bh");

  // build graph
  init(x, h);
}

GRU::GRU(Graph& graph, Function& x, Function& h, const GRU& other) :
Function(graph)
{
  // share variables with the "other"
  _Wz = other._Wz;
  _Uz = other._Uz;
  _bz = other._bz;

  _Wr = other._Wr;
  _Ur = other._Ur;
  _br = other._br;

  _Wh = other._Wh;
  _Uh = other._Uh;
  _bh = other._bh;

  // build graph
  init(x, h);
}

const Tensor& GRU::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // create value h(t)
  _value = _GRU->forward();

  // return value
  return _value;
}

void GRU::init(Function& x, Function& h)
{
  iforward(&x);
  iforward(&h);

  // z
  auto& z = *_graph.new_sigmoid(
    product(x,*_Wz) + product(h,*_Uz) + *_bz);

  // r
  auto& r = *_graph.new_sigmoid(
    product(x,*_Wr) + product(h,*_Ur) + *_br);

  // c
  auto& c = *_graph.new_tanh(
    product(x,*_Wh) + product(r * h,*_Uh) + *_bh);

  // h(t)
  _GRU = &(z * h + (1-z) * c);

  // let output handle the gradient directly
  _GRU->ibackward(_graph.new_iderivative(*this));
}

///////////////////////////////////////////
// Function LSTM
///////////////////////////////////////////

LSTM::LSTM(Graph& graph, Function& x, Function& h, Function& c, int in, int out) :
Function(graph)
{
  // construct new variables (row-major)
  _Wxi = graph.new_variable(in, out, "Wxi");
  _Whi = graph.new_variable(out, out, "Hhi");
  _Wci = graph.new_variable(out, out, "Hci");
  _bi = graph.new_variable(1, out, "bi");

  _Wxf = graph.new_variable(in, out, "Wxf");
  _Whf = graph.new_variable(out, out, "Hhf");
  _Wcf = graph.new_variable(out, out, "Hcf");
  _bf = graph.new_variable(1, out, "bf");

  _Wxc = graph.new_variable(in, out, "Wxc");
  _Whc = graph.new_variable(out, out, "Whc");
  _bc = graph.new_variable(1, out, "bc");

  _Wxo = graph.new_variable(in, out, "Wxo");
  _Who = graph.new_variable(out, out, "Who");
  _Wco = graph.new_variable(out, out, "Wco");
  _bo = graph.new_variable(1, out, "bo");

  // build graph
  init(x, h, c);
}

LSTM::LSTM(Graph& graph, Function& x, Function& h, Function& c, const LSTM& other) :
Function(graph)
{
  // share variables with the "other"
  _Wxi = other._Wxi;
  _Whi = other._Whi;
  _Wci = other._Wci;
  _bi = other._bi;

  _Wxf = other._Wxf;
  _Whf = other._Whf;
  _Wcf = other._Wcf;
  _bf = other._bf;

  _Wxc = other._Wxc;
  _Whc = other._Whc;
  _bc = other._bc;

  _Wxo = other._Wxo;
  _Who = other._Who;
  _Wco = other._Wco;
  _bo = other._bo;

  // build graph
  init(x, h, c);
}

const Tensor& LSTM::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // create value h(t)
  _value = _LSTM->forward();

  // return value
  return _value;
}

// Long Short-Term Memory (LSTM) function
// https://arxiv.org/pdf/1308.0850
// i(t) = Sigmoid(Wxi * x(t) + Whi * h(t-1) + Wci * c(t-1) + bi)
// f(t) = Sigmoid(Wxf * x(t) + Whf * h(t-1) + Whi * c(t-1) + bh)
// c(t) = f(t) . c(t-1) + i(t) . Tanh(Wxc * x(t) + Whc * h(t-1) + bc)
// o(t) = Sigmoid(Wxo * x(t) + Who * h(t-1) + Who * c(t) + bo)
// h(t) = o(t) . Tanh(c(t))
void LSTM::init(Function& x, Function& h, Function& c)
{
  iforward(&x);
  iforward(&h);
  iforward(&c);

  // i
  auto& i = *_graph.new_sigmoid(
    product(x,*_Wxi) + product(h,*_Whi) + product(c,*_Wci) + *_bi);

  // f
  auto& f = *_graph.new_sigmoid(
    product(x,*_Wxf) + product(h,*_Whf) + product(c,*_Wcf) + *_bf);

  // c
  auto& new_c = f * c + i * *_graph.new_tanh(
    product(x,*_Wxc) + product(h,*_Whc) + *_bc);

  // o
  auto& o = *_graph.new_sigmoid(
    product(x,*_Wxo) + product(h,*_Who) + product(new_c,*_Wco) + *_bo);

  // h
  auto& new_h = o * *_graph.new_tanh(new_c);

  // set references
  _cell = &new_c;
  _LSTM = &new_h;

  // let output handle the gradient directly
  _LSTM->ibackward(_graph.new_iderivative(*this));
}

///////////////////////////////////////////
// Norm
///////////////////////////////////////////

Norm::Norm(Graph& graph, Function& x, int rows, int cols, DTYPE eps) :
Function(graph), _epsilon(eps)
{
  if (rows * cols > 1)
  {
    _a = graph.new_variable(rows, cols, "Norm.A");
    _b = graph.new_variable(rows, cols, "Norm.B");
  }
  else
  {
    _a = graph.new_variable(1, 1, "Norm.a");
    _b = graph.new_variable(1, 1, "Norm.b");
  }

  _a->value() = Tensor::Ones(_a->value().rows(), _a->value().cols());
  _b->value() = Tensor::Zero(_b->value().rows(), _b->value().cols());

  init(x);
}

Norm::Norm(Graph& graph, Function& x, const Norm& other) :
Function(graph), _epsilon(other._epsilon)
{
  _a = other._a;
  _b = other._b;

  init(x);
}

void Norm::init(Function& x)
{
  iforward(&x);

  _H = _graph.new_constant(1,1); // dimension placeholder

  auto& mean = *_graph.new_sum(x) / *_H;
  auto& x_mean = x - *_graph.new_broadcast(mean, x);

  auto& var = *_graph.new_sum(x_mean * x_mean) / *_H;
  auto& std = power(var + _epsilon, 0.5);

  _N = &(x_mean / *_graph.new_broadcast(std, x_mean));

  if (_a->value().size() > 1 && _b->value().size() > 1)
  {
    _N = &(*_N * *_a + *_b);
  }
  else
  {
    auto a = _graph.new_broadcast(*_a, *_N);
    auto b = _graph.new_broadcast(*_b, *_N);

    _N = &(*_N * *a + *b);
  }

  _N->ibackward(_graph.new_iderivative(*this));
}

// F = a * (x - m) / s - b
const Tensor& Norm::forward()
{
  // return cached value
  if (_value.size()) return _value;

  auto& x = _forward[0]->forward();

  // update dimension placeholder
  _H->value() = Tensor::Constant(1,1, x.rows() * x.cols());

  // compute norm
  _value = _N->forward();

  return _value;
}

///////////////////////////////////////////
// Sampler
///////////////////////////////////////////

Sampler::Sampler(Graph& graph, Function& m, Function& s) :
Function(graph), _enabled(true)
{
  iforward(&m);
  iforward(&s);

  // random sampling with parametrization trick for back-propagation
  // Auto-Encoding Variational Bayes by Diederik P. Kingma, Max Welling
  // https://arxiv.org/pdf/1312.6114.pdf

  _e = _graph.new_constant();

  _Z = &(m + *_e * s);

  _Z->ibackward(_graph.new_iderivative(*this));
}

const Tensor& Sampler::forward()
{
  // return cached value
  if (_value.size()) return _value;

  auto& m = _forward[0]->forward();

  int rows = m.rows();
  int cols = m.cols();

  // sample random constant
  if (_enabled)
  {
    auto random = [this]() { return _graph.random().normal_dec(0, 1); };
    _e->value() = Tensor::NullaryExpr(rows, cols, random);
  }
  else
  {
    _e->value() = Tensor::Zero(rows, cols);
  }

  // update value
  _value = _Z->forward();

  // return value
  return _value;
}

///////////////////////////////////////////
// Gaussian
///////////////////////////////////////////

Gaussian::Gaussian(Graph& graph, Function& x, Function& m, Function& s) :
Function(graph)
{
  iforward(&x);
  iforward(&m);
  iforward(&s);

  // compute height of distribution's peak
  _a = &(1 / (s * sqrt(2 * M_PI)));

  // compute argument for exp(z)
  _z = &(-0.5 * power((x - m) / s, 2));

  // for performance and numerical stability we do NOT add exp(z) to the graph,
  // instead, we calculate exp(z) in forward() method and use
  // F = a * exp(z), with dFdz = a * exp(z), dFda = exp(z)

  // Derivative with respect to a
  class Derivative_a : public Function
  {
  public:
    Derivative_a(Graph& graph, Gaussian& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._z);
      graph.keep(this);
    }

    // dFda = exp(z)
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& z = _forward[1]->forward();

      auto dFda = z.array().exp();

      // update gradient value
      _value = g.array() * dFda;

      // return gradient value
      return _value;
    }
  };

  // Derivative with respect to z
  class Derivative_z : public Function
  {
  public:
    Derivative_z(Graph& graph, Gaussian& base) : Function(graph)
    {
      iforward(&base);
      graph.keep(this);
    }

    // dFdz = a * exp(z) = F
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& dFdz = _forward[0]->forward();

      // update gradient value
      _value = g.array() * dFdz.array();

      // return gradient value
      return _value;
    }
  };

  _a->ibackward(new Derivative_a(_graph, *this));
  _z->ibackward(new Derivative_z(_graph, *this));
}

// A = 1 / (s * sqrt(2 * PI))
// Z = -(x - m)^2 / (2*s^2)
// F = A * exp(Z)
const Tensor& Gaussian::forward()
{
  // return cached value
  if (_value.size()) return _value;

  auto& a = _a->forward();
  auto& z = _z->forward();

  // F = a * exp(Z)
  _value = a.array() * z.array().exp();

  // return value
  return _value;
}

///////////////////////////////////////////
// Log-Gaussian
///////////////////////////////////////////

LogGaussian::LogGaussian(Graph& graph, Function& x, Function& m, Function& s) :
Function(graph)
{
  iforward(&x);
  iforward(&m);
  iforward(&s);

  // compute height of distribution's peak
  _a = &(1 / (s * sqrt(2 * M_PI)));

  // compute argument for exp(z)
  _z = &(-0.5 * power((x - m) / s, 2));

  // for performance and numerical stability we do NOT add exp(z) to the graph,
  // instead, we calculate exp(z) in forward() method and use
  // F = log(a * exp(z)) = log(a) + z, with dFdz = 1, dFda = 1/a

  // Derivative with respect to a
  class Derivative_a : public Function
  {
  public:
    Derivative_a(Graph& graph, LogGaussian& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._a);
      graph.keep(this);
    }

    // dFda = exp(z)
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& a = _forward[1]->forward();
      auto dFda = 1 / a.array();

      // update gradient value
      _value = g.array() * dFda;

      // return gradient value
      return _value;
    }
  };

  // Derivative with respect to z
  class Derivative_z : public Function
  {
  public:
    Derivative_z(Graph& graph, LogGaussian& base) : Function(graph)
    {
      iforward(&base);
      graph.keep(this);
    }

    // dFdz = a * exp(z) = F
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      // auto dFdz = Tensor::Ones(g.rows(), g.cols());

      // update gradient value
      _value = g.array() /** dFdz.array()*/;

      // return gradient value
      return _value;
    }
  };

  _a->ibackward(new Derivative_a(_graph, *this));
  _z->ibackward(new Derivative_z(_graph, *this));
}

// A = 1 / (s * sqrt(2 * PI))
// Z = -(x - m)^2 / (2*s^2)
// F = log(A * exp(Z)) = log(A) + Z
const Tensor& LogGaussian::forward()
{
  // return cached value
  if (_value.size()) return _value;

  auto& a = _a->forward();
  auto& z = _z->forward();

  _value = a.array().log() + z.array();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Embedding
///////////////////////////////////////////

Embedding::Embedding(Graph& graph, Function& index, int in, int out) :
Function(graph)
{
  // construct new variables with row-wise embedding vectors
  _E = graph.new_variable(in, out, "Embedding.E");

  init(index);
}

Embedding::Embedding(Graph& graph, Function& index, const Embedding& other) :
Function(graph)
{
  // share variables with the "other"
  _E = other._E;

  init(index);
}

void Embedding::init(Function& index)
{
  iforward(&index);

  // Derivative with respect to E
  class Derivative_E : public Function
  {
  public:
    Derivative_E(Graph& graph, Embedding& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._E);
      iforward(base._forward[0]);
      graph.keep(this);
    }

    // dFdE = x(i)
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _forward[0]->backward();
      auto& E = _forward[1]->forward();
      auto& index = _forward[2]->forward();

      // update gradient value
      _value = Tensor::Zero(E.rows(), E.cols());

      for (int i=0; i<index.size(); i++)
      {
        _value.row((size_t)index(i)) = g.row(i);
      }

      return _value;
    }
  };

  _E->ibackward(new Derivative_E(_graph, *this));
}

// F = E * x(i)
const Tensor& Embedding::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get variables
  auto& E = _E->forward();
  auto& index = _forward[0]->forward();

  // create initial value
  _value = Tensor::Zero(index.size(), E.cols());

  for (int i=0; i<index.size(); i++)
  {
    _value.row(i) = E.row((size_t)index(i));
  }

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Conv2D
///////////////////////////////////////////

Conv2D::Conv2D(
  Graph& graph,
  Function& x,
  int i_rows,
  int i_cols,
  int i_channels,
  int o_channels,
  int k_rows,
  int k_cols,
  int stride,
  int padding,
  int dilation) :
Function(graph),
_i_rows(i_rows),
_i_cols(i_cols),
_i_channels(i_channels),
_o_channels(o_channels),
_k_rows(k_rows),
_k_cols(k_cols),
_stride(stride),
_padding(padding),
_dilation(dilation)
{
  // Kernels for input and output channels
  //
  // x - flat 2D col-major matrix
  // I - number of input channels
  // O - number of output channels
  // X_i - channel i of input X
  // Y_o - channel o of output Y
  // K_i_o - kernel [i,o] of input X_i and output Y_o
  // K_r - number of kernel rows
  // K_c - number of kernel cols
  //
  // D = [K_r, K_c], kernel dimentions
  // N = O * I, numbef of kernels
  // E = K_r * K_c, number of single kernel parameters
  // A = N * E, number of all kernel parameters
  //
  // Kernels are stored in kernel matrix as sub-matrices or blocks.
  //
  // Kernel Matrix K = [O * K_r, I * K_c]
  //
  // [K_1_1,...,K_I_1] // kernels for output channel 0
  // .................
  // [K_1_O,...,K_I_O] // kernels for outout channel O-1
  //
  // Examples
  //
  // Pseudo code to access kernel for input "i" and output "o":
  //
  // K[o * K_r, i * K_c, K_r, K_c]
  //
  // Eigen code to access kernel for input "i" and output "o" as matrix:
  //
  // auto kernel_i_o = K.block(o * K_r, i * K_c, K_r, K_c);
  //
  // Eigen code to access kernel for input "i" and output "o" as vector:
  //
  // auto kernel_vector_i_o = ConstTensorMap(kernel_i_o.data(), E, 1);
  //

  // construct new variables
  _K = graph.new_variable(_o_channels * _k_rows, _i_channels * _k_cols);

  init(x);
}

Conv2D::Conv2D(Graph& graph, Function& x, const Conv2D& other) :
Function(graph)
{
  // share dimentions and variables with the "other"
  _i_rows = other._i_rows;
  _i_cols = other._i_cols,
  _i_channels = other._i_channels,
  _o_channels = other._o_channels,
  _k_rows = other._k_rows;
  _k_cols = other._k_cols;
  _stride = other._stride,
  _padding = other._padding,
  _dilation = other._dilation,
  _K = other._K;

  init(x);
}

const Tensor& Conv2D::forward()
{
  // return cached value
  if (_value.size()) return _value;

  auto& x = _forward[0]->forward();
  auto& K = K_matrix();

  //_value = K * x; // col major
  _value = ABT(x,K); // row major

  // return value
  return _value;
}

void Conv2D::init(Function& x)
{
  iforward(&x);

  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Conv2D& base) : Function(graph)
    {
      iforward(&base);
      graph.keep(this);
    }

    // dFdx = K_matrix
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto base = (Conv2D*)_forward[0];

      auto& g = base->backward();
      auto& K = base->K_matrix();

      auto& dFdx = K;

      // update gradient value
      //_value = ABT(g, dFdx); // col major
      _value = g * dFdx; // row major

      return _value;
    }
  };

  // Derivative with respect to K
  class Derivative_K : public Function
  {
  public:
    Derivative_K(Graph& graph, Conv2D& base) : Function(graph)
    {
      iforward(&base);
      iforward(base._forward[0]);
      graph.keep(this);
    }

    // dFdK = x
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto base = (Conv2D*)_forward[0];
      auto base_x = _forward[1];

      auto& g = base->backward();
      auto& x = base_x->forward();
      auto& K = base->K_matrix();

      auto& dFdK = x;
      //auto dFdK_matrix = ABT(g, dFdK, K); // col major
      auto dFdK_matrix = ATB(g, dFdK, K); // row major

      // update gradient value
      _value = base->K_gradient(dFdK_matrix);

      return _value;
    }
  };

  x.ibackward(new Derivative_x(_graph, *this));
  _K->ibackward(new Derivative_K(_graph, *this));
}

SparseTensor& Conv2D::K_matrix()
{
  if (_K_tracker.size() && _K_tracker == _K->value()) return _K_matrix;

  _K_tracker = _K->value();

  // convert matrix K to unrolled matrix K_matrix
  convert(_K->value(), _K_matrix, true);

  return _K_matrix;
}

Tensor Conv2D::K_gradient(SparseTensor& dK_matrix)
{
  // K gradient
  Tensor dK = Tensor::Zero(
    _o_channels * _k_rows,
    _i_channels * _k_cols
  );

  // convert unrolled gradient dK_matrix to gradient dK
  convert(dK, dK_matrix, false);

  return dK;
}

void Conv2D::convert(Tensor& K, SparseTensor& K_matrix, bool forward)
{
  // build kernel matrix by sliding kernel over input
  //
  // Input for single channel
  //
  // 1 2 3
  // 4 5 6
  //
  // Multiple channels are in plannar format (channel, row, col)
  //
  // R,R,R, G,G,G, B,B,B
  //
  // Inputs with channels in packed format (row, col, channel)
  //
  // R,G,B, R,G,B, R,G,B
  //
  // should be converted to plannar format
  //
  // Padded input
  //
  // 0 0 0 0 0
  // 0 1 2 3 0
  // 0 4 5 6 0
  // 0 0 0 0 0
  //
  // Kernel
  //
  // 10 20
  // 30 40
  //
  // Kernel with dilation
  //
  // 10 0 20
  //  0 0  0
  // 30 0 40
  //
  // Convolution over all input positions
  //
  //   10 20 30 40
  // 1  0  0  0  1
  // 2  0  0  1  2
  // 3  0  0  2  3
  // 4  0  0  3  0
  // 5  0  1  0  4
  // 6  1  2  4  5
  // 7  2  3  5  6
  // 8  3  0  6  0
  // 9  0  4  0  0
  // 0  4  5  0  0
  // 1  5  6  0  0
  // 2  6  0  0  0
  //
  // Resulting kernel matrix for one channel
  //
  //     1  2  3  4  5  6
  // 1  40
  // 2  30 40
  // 3     30 40
  // 4        30
  // 5  20       40
  // 6  10 20    30 40
  // 7     10 20    30 40
  // 8        10       30
  // 9           20
  // 0           10 20
  // 1              10 20
  // 2                 10
  //
  //
  // Resulting kernel matrix for mulitple channels
  //
  //     input channel 1      input channel I
  //
  //     1  2  3  4  5  6 ... 1  2  3  4  5  6    o
  // 1  40                ... 40                  u
  // 2  30 40             ... 30 40               t
  // 3     30 40          ...    30 40
  // 4        30          ...       30            c
  // 5  20       40       ... 20       40         h
  // 6  10 20    30 40    ... 10 20    30 40      a
  // 7     10 20    30 40 ...    10 20    30 40   n
  // 8        10       30 ...       10       30   n
  // 9           20       ...          20         e
  // 0           10 20    ...          10 20      l
  // 1              10 20 ...             10 20
  // 2                 10 ...                10   1
  //    .......................................
  //    .......................................
  //    .......................................
  //     1  2  3  4  5  6 ... 1  2  3  4  5  6    o
  // 1  40                ... 40                  u
  // 2  30 40             ... 30 40               t
  // 3     30 40          ...    30 40
  // 4        30          ...       30            c
  // 5  20       40       ... 20       40         h
  // 6  10 20    30 40    ... 10 20    30 40      a
  // 7     10 20    30 40 ...    10 20    30 40   n
  // 8        10       30 ...       10       30   n
  // 9           20       ...          20         e
  // 0           10 20    ...          10 20      l
  // 1              10 20 ...             10 20
  // 2                 10 ...                10   O

  // row / column dilation
  int r_d = _dilation;
  int c_d = _dilation;

  // row / column padding
  int r_p = _padding;
  int c_p = _padding;

  // row / column stride
  int r_s = _stride;
  int c_s = _stride;

  // input with padding
  int i_padded_rows = _i_rows + 2 * r_p;
  int i_padded_cols = _i_cols + 2 * c_p;

  // kernel with dilation
  int k_span_rows = r_d * (_k_rows - 1) + 1;
  int k_span_cols = c_d * (_k_cols - 1) + 1;

  // output
  int o_rows = (_i_rows - k_span_rows + 2 * r_p) / r_s + 1;
  int o_cols = (_i_cols - k_span_cols + 2 * c_p) / c_s + 1;

  // kernel matrix for input and output channels
  K_matrix.conservativeResize(
    _o_channels * o_rows * o_cols,
    _i_channels *_i_rows * _i_cols
  );

  // input mask with padding
  TensorXi i_mask = TensorXi::Zero(i_padded_rows, i_padded_cols);
  i_mask.block(r_p, c_p, _i_rows, _i_cols) = TensorXi::Ones(_i_rows, _i_cols);

  // kernel mask with dilation
  TensorXi k_mask = TensorXi::Ones(k_span_rows, k_span_cols);
  if (_dilation > 1)
  {
    for (int r=0; r<_k_rows - 1; r++)
    {
      k_mask.block(r * r_d + 1, 0, r_d - 1, k_span_cols).array() = 0;
    }
    for (int c=0; c<_k_cols - 1; c++)
    {
      k_mask.block(0, c * c_d + 1, k_span_rows, c_d - 1).array() = 0;
    }
  }

  // update kernel matrix for all input and output channels
  for (int i=0; i < _i_channels; i++)
  for (int o=0; o < _o_channels; o++)
  {
    auto kernel = K.block(
      o * _k_rows,
      i * _k_cols,
      _k_rows,
      _k_cols
    );

    auto k_matrix = K_matrix.block(
      o * o_rows * o_cols,
      i *_i_rows * _i_cols,
      o_rows * o_cols,
      _i_rows * _i_cols
    );

    // row major
    for (int r=0, m_r=0; r <= i_padded_rows - k_span_rows; r++)
    for (int c=0; c <= i_padded_cols - k_span_cols; c++, m_r++)
    // col major
    //for (int c=0, m_r=0; c <= i_padded_cols - k_span_cols; c++)
    //for (int r=0; r <= i_padded_rows - k_span_rows; r++, m_r++)
    {
      TensorXi conv = i_mask.block(r, c, k_span_rows, k_span_cols);
      conv.array() *= k_mask.array();

      for (int k_r=0; k_r < _k_rows; k_r++)
      for (int k_c=0; k_c < _k_cols; k_c++)
      {
        // convert kernel index to convolution index
        int conv_r = k_r * r_d;
        int conv_c = k_c * c_d;

        if (conv(conv_r, conv_c))
        {
          // convert input coordinates to kernel matrix column (row major)
          int m_c = (r - r_p + conv_r) * _i_cols + (c - c_p + conv_c);
          // convert input coordinates to kernel matrix column (col major)
          // int m_c = (r - r_p + conv_r) + (c - c_p + conv_c) * _i_rows;
          if (forward)
          {
            k_matrix.coeffRef(m_r, m_c) = kernel(k_r, k_c);
          }
          else
          {
            kernel(k_r, k_c) += k_matrix.coeff(m_r, m_c);
          }
        }
      }
    }
  }
}

///////////////////////////////////////////
// Graph
///////////////////////////////////////////

void Graph::clear()
{
  for (auto e: _nodes) delete e;
  _nodes.clear();
  _vars.clear();
}

// get function by name
Function* Graph::function(const std::string& name) const
{
  for (auto i=0; i<_nodes.size(); i++)
  {
    if (_nodes[i]->name() == name) return _nodes[i];
  }

  return nullptr;
}

// get nambed variables
std::vector<Variable*> Graph::named_variables() const
{
  std::vector<Variable*> vars;

  for (auto v: _vars)
  {
    if (v->name().size()) vars.push_back(v);
  }

  return vars;
}

// track function
void Graph::keep(Function* f, const char* name)
{
  _nodes.push_back(f);
  if (name) f->name(name);
}

// track variable
void Graph::keep(Variable* v, const char* name)
{
  _nodes.push_back(v);
  _vars.push_back(v);
  if (name) v->name(name);
}

// reset cache
void Graph::recache()
{
  for (auto e: _nodes) e->recache();
}

// reset gradients
void Graph::zero_grad()
{
  for (auto e: _vars) e->gradient().setZero();
}

// compute values
const Tensor& Graph::forward(Function& f)
{
  struct ExecInfo
  {
    size_t users;
    size_t depth;
  };

  // execution plan
  std::unordered_map<Function*, ExecInfo> plan;

  // add top level node to execution plan
  ExecInfo info = {0, 0};
  plan.emplace(&f, info);

  // add top level node to node stack
  std::vector<Function*> stack;
  stack.push_back(&f);

  // collect dependent nodes
  while(stack.size())
  {
    auto node = stack.back();
    auto& info = plan[node];
    stack.pop_back();

    for (auto input: node->iforward())
    {
      auto it = plan.find(input);
      if (it != plan.end())
      {
        it->second.users++;
        it->second.depth = std::max(it->second.depth, info.depth+1);
      }
      else
      {
        ExecInfo info = {1, info.depth+1};
        plan.emplace(input, info);
      }
      stack.push_back(input);
    }
  }

  // create list of nodes to execute
  for (auto it=plan.begin(); it != plan.end(); it++)
  {
    stack.push_back(it->first);
  }

  // sort nodes by node depth in graph
  std::sort(stack.begin(), stack.end(), [&plan](Function* a, Function* b)
  {
    auto& info_a = plan[a];
    auto& info_b = plan[b];
    return info_a.depth > info_b.depth;
  });

  // execute nodes in order of graph depth (from bottom to top)
  for (auto n: stack)
  {
    auto& info = plan[n];

    // execute node
    n->forward();

    // decrement usage in input nodes and recache when no longer needed
    for (auto input: n->iforward())
    {
      auto& info = plan[input];
      info.users--;
      if (info.users == 0)
      {
        input->recache();
      }
    }
  }

  // return the cached f value
  return f.forward();
}

// compute gradients
void Graph::backward(Function& f, const Tensor& g)
{
  f.forward();
  f.gradient() = g;
  for (auto e: _vars) e->backward();
}

///////////////////////////////////////////
// numerical derivative
///////////////////////////////////////////

DTYPE Graph::dFdX(Function& f, Variable& x, int fr, int fc, int xr, int xc)
{
  auto& v = x.value();
  DTYPE x0 = v(xr,xc);

  // F(x + h)
  recache();
  v(xr,xc) = x0 + FINITE_DELTA;
  DTYPE f2 = f.forward()(fr,fc);

  // F(x - h)
  recache();
  v(xr,xc) = x0 - FINITE_DELTA;
  DTYPE f1 = f.forward()(fr,fc);

  // restore x
  v(xr,xc) = x0;

  // (F(x + h) - F(x - h)) / 2h
  return (f2 - f1) / (2 * FINITE_DELTA);
}

Tensor Graph::dFdX(Function& f, Variable& x)
{
  // get provided gradient and its size
  Tensor df = f.gradient();
  auto size = df.size();

  int f_rows = f.forward().rows();
  int f_cols = f.forward().cols();
  int x_rows = x.forward().rows();
  int x_cols = x.forward().cols();

  // set default gradient value to 1, if not provided
  if (!size)
  {
    df = Tensor::Ones(f_rows, f_cols);
  }

  Tensor dfdx = Tensor::Zero(x_rows, x_cols);

  for (int fr=0; fr<f_rows; fr++)
  for (int fc=0; fc<f_cols; fc++)
  for (int xr=0; xr<x_rows; xr++)
  for (int xc=0; xc<x_cols; xc++)
  dfdx(xr, xc) = dfdx(xr, xc) + dFdX(f, x, fr, fc, xr, xc) * df(fr, fc);

  // reset gradient to the provided value
  if (size)
  {
    f.gradient() = df;
  }

  return dfdx;
}

///////////////////////////////////////////
// node name scope
///////////////////////////////////////////

std::string Graph::scope_name() const
{
  int size = _scope.size();
  for (const auto& e: _scope) size += e.size();

  std::string name;
  name.reserve(size);

  for (const auto& e: _scope)
  {
    name += e;
    name += ':';
  }

  return name;
}

///////////////////////////////////////////
// Default gradient aggregator
///////////////////////////////////////////

void Graph::aggregate(Tensor& g, const std::vector<Function*>& derivative) const
{
  // aggregate gradient
  for (auto e: derivative)
  {
    try
    {
      g += e->forward();
    }
    catch(NoValueException& e)
    {
      continue;
    }
  }
}

} /* namespace */

