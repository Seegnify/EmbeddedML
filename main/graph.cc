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

    if (_backprop) _graph.aggregate(_gradient, _derivative);
  }

  return _gradient;
}

// recache function value and gradient
void Function::recache()
{
  _value.resize(0,0);
  _gradient.resize(0,0);
}

///////////////////////////////////////////
// Identity Derivarive impl
///////////////////////////////////////////

IDerivative::IDerivative(Graph& graph, Function& base) :
Function(graph), _base(base) {}

// dFdx = base.dFdx
const Tensor& IDerivative::forward()
{
  // return cached value
  if (_value.size()) return _value;

  auto& g = _base.backward();

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

  _y->derivative(graph.new_iderivative(*this));
}

Rowwise::Rowwise(Graph& graph, Function& x, int rows, int cols,
std::function<Function*(Function& block)> shared_ctor,
std::function<Function*(Function& block, Function& shared)> ctor) :
Function(graph)
{
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

  _y->derivative(graph.new_iderivative(*this));
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
std::function<Function*(Function& block)> ctor) : Function(graph)
{
  // transpose and apply row-wise ctor, then transpose again
  _y = graph.new_transpose(
    *graph.new_rowwise(
      *graph.new_transpose(x), rows, cols, ctor
    )
  );

  _y->derivative(graph.new_iderivative(*this));
}

Colwise::Colwise(Graph& graph, Function& x, int rows, int cols,
std::function<Function*(Function& block)> shared_ctor,
std::function<Function*(Function& block, Function& shared)> ctor) :
Function(graph)
{
  // transpose and apply row-wise ctor, then transpose again
  _y = graph.new_transpose(
    *graph.new_rowwise(
      *graph.new_transpose(x), rows, cols, shared_ctor, ctor
    )
  );

  _y->derivative(graph.new_iderivative(*this));
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

Constant::Constant(Graph& graph, int rows, int cols) : 
Function(graph)
{
  _value.resize(rows, cols);
}

///////////////////////////////////////////
// Function Variable
///////////////////////////////////////////

// random variable
Variable::Variable(Graph& graph, int rows, int cols) : 
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

  if (_backprop) _graph.aggregate(_gradient, _derivative);

  return _gradient;
}

///////////////////////////////////////////
// Function Broadcast
///////////////////////////////////////////

Broadcast::Broadcast(Graph& graph, Function& x, Function& target) :
Function(graph), _x(x), _t(target)
{
  // Broadcast Backward function
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Broadcast& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = 1
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& x = _base._x.forward();

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

  private:
    Broadcast& _base;
  };

  _x.derivative(new Derivative_x(_graph, *this));
}

// F = x (x broadcast to target)
const Tensor& Broadcast::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get source and target
  auto& x = _x.forward();
  auto& t = _t.forward();

  // broadcast row-wise
  if (x.rows() == 1 && x.cols() > 1)
  {
    _value = Tensor::Zero(t.rows(), t.cols());
    _value.rowwise() += ConstRowVectorMap(x.data(), x.size());
  }
  // broadcast col-wise
  else if (x.rows() > 1 && x.cols() == 1)
  {
    _value = Tensor::Zero(t.rows(), t.cols());
    _value.colwise() += ConstColVectorMap(x.data(), x.size());
  }
  // broadcast any size
  else
  {
    _value = Tensor::Constant(t.rows(), t.cols(), x.sum());
  }

  return _value;
}

///////////////////////////////////////////
// Function Reshape
///////////////////////////////////////////

Reshape::Reshape(Graph& graph, Function& x, int rows, int cols) :
Function(graph), _x(x), _rows(rows), _cols(cols)
{
  // Reshape Backward function
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Reshape& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = 1
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& x = _base._x.forward();

      // pass the reshaped gradient through (default is ColMajor)
      auto dFdx = g.reshaped<Eigen::RowMajor>(x.rows(), x.cols());

      // update gradient value
      _value = dFdx;

      // return gradient value
      return _value;
    }

  private:
    Reshape& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
}

// F = x
const Tensor& Reshape::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _x.forward();

  // update value (reshaped() default is ColMajor)
  _value = x.reshaped<Eigen::RowMajor>(_rows, _cols);

  return _value;
}

///////////////////////////////////////////
// Function Split
///////////////////////////////////////////

Split::Split(Graph& graph, Function& x, int r, int c, int rows, int cols) :
Function(graph), _x(x), _r(r), _c(c), _rows(rows), _cols(cols)
{
  // Split Backward function
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Split& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = 1
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& x = _base._x.forward();

      // pass the gradient through the block
      Tensor dFdx = Tensor::Zero(x.rows(), x.cols());
      dFdx.block(_base._r, _base._c, _base._rows, _base._cols) = g;

      // update gradient value
      _value = dFdx;

      // return gradient value
      return _value;
    }    

  private:
    Split& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
}

// F = block(x)
const Tensor& Split::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _x.forward();

  // update value
  _value = x.block(_r, _c, _rows, _cols);

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Join
///////////////////////////////////////////

Join::Join(Graph& graph, Function& x, Function& y, int rows, int cols) :
Function(graph), _x(x), _y(y), _rows(rows), _cols(cols)
{
  // Join Derivative with respect to X
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Join& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = 1
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& x = _base._x.forward();

      // split g into gx and gy (g data is already flat)
      ConstTensorMap gx(g.data(), x.rows(), x.cols());

      // update gradient value
      _value = gx;
      return _value;
    }    

  private:
    Join& _base;
  };

  // Join Derivative with respect to Y
  class Derivative_y : public Function
  {
  public:
    Derivative_y(Graph& graph, Join& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdy = 1
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& x = _base._x.forward();
      auto& y = _base._y.forward();

      // split g into gx and gy (g data is already flat)
      ConstTensorMap gy(g.data() + x.size(), y.rows(), y.cols());

      // update gradient value
      _value = gy;
      return _value;
    }    

  private:
    Join& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
  _y.derivative(new Derivative_y(graph, *this));
}

// F = join(x)
const Tensor& Join::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _x.forward();
  auto& y = _y.forward();

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

Min::Min(Graph& graph, Function& x, Function& y) :
Function(graph), _x(x), _y(y)
{
  // Derivative with respect to X
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Min& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = 1 if x < y, 0 otherwise
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& x = _base._x.forward();
      auto& y = _base._y.forward();

      // compute gradient mask
      auto one = Tensor::Ones(x.rows(), x.cols());
      auto zero = Tensor::Zero(x.rows(), x.cols());
      auto dFdx_mask = (x.array() < y.array()).select(one, zero);
      auto dFdy_mask = (x.array() > y.array()).select(one, zero);

      // update gradient value
      _value = g.array() * dFdx_mask.array();
      return _value;
    }    

  private:
    Min& _base;
  };

  // Derivative with respect to Y
  class Derivative_y : public Function
  {
  public:
    Derivative_y(Graph& graph, Min& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdy = 1 if x > y, 0 otherwise
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& x = _base._x.forward();
      auto& y = _base._y.forward();

      // compute gradient mask
      auto one = Tensor::Ones(x.rows(), x.cols());
      auto zero = Tensor::Zero(x.rows(), x.cols());
      auto dFdx_mask = (x.array() < y.array()).select(one, zero);
      auto dFdy_mask = (x.array() > y.array()).select(one, zero);

      // update gradient value
      _value = g.array() * dFdy_mask.array();
      return _value;
    }    

  private:
    Min& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
  _y.derivative(new Derivative_y(graph, *this));
}

// F = min(x,y)
const Tensor& Min::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _x.forward();
  auto& y = _y.forward();

  // update value
  _value = x.array().min(y.array());

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Max
///////////////////////////////////////////

Max::Max(Graph& graph, Function& x, Function& y) :
Function(graph), _x(x), _y(y)
{
  // Derivative with respect to X
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Max& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = 1 if x > y, 0 otherwise
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& d = _base.backward();
      auto& x = _base._x.forward();
      auto& y = _base._y.forward();

      // compute gradient mask
      auto one = Tensor::Ones(x.rows(), x.cols());
      auto zero = Tensor::Zero(x.rows(), x.cols());
      auto dFdx_mask = (x.array() > y.array()).select(one, zero);

      // update gradient value
      _value = d.array() * dFdx_mask.array();
      return _value;
    }    

  private:
    Max& _base;
  };

  // Derivative with respect to Y
  class Derivative_y : public Function
  {
  public:
    Derivative_y(Graph& graph, Max& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdy = 1 if x < y, 0 otherwise
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& d = _base.backward();
      auto& x = _base._x.forward();
      auto& y = _base._y.forward();

      // compute gradient mask
      auto one = Tensor::Ones(x.rows(), x.cols());
      auto zero = Tensor::Zero(x.rows(), x.cols());
      auto dFdy_mask = (x.array() < y.array()).select(one, zero);

      // update gradient value
      _value = d.array() * dFdy_mask.array();
      return _value;
    }    

  private:
    Max& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
  _y.derivative(new Derivative_y(graph, *this));
}

// F = max(x,y)
const Tensor& Max::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _x.forward();
  auto& y = _y.forward();

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

  _y->derivative(_graph.new_iderivative(*this));
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
Function(graph), _x(x), _y(y)
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Product& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = y.T
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& y = _base._y.forward();

      auto& dFdx = y;

      // update gradient value
      _value = ABT(g, dFdx);
      return _value;
    }    

  private:
    Product& _base;
  };

  // Derivative with respect to y
  class Derivative_y : public Function
  {
  public:
    Derivative_y(Graph& graph, Product& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdy = x.T
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& x = _base._x.forward();

      auto& dFdy = x;

      // update gradient value
      _value = ATB(dFdy, g);
      return _value;
    }    

  private:
    Product& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
  _y.derivative(new Derivative_y(graph, *this));
}

// F = x * y
const Tensor& Product::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _x.forward();
  auto& y = _y.forward();

  // create cached value
  _value.noalias() = x * y;

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Add
///////////////////////////////////////////

Add::Add(Graph& graph, Function& x, Function& y) : 
Function(graph), _x(x), _y(y) 
{
  // Derivative with respect to x or y
  class Derivative_any : public Function
  {
  public:
    Derivative_any(Graph& graph, Add& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = 1
    // dFdy = 1
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();

      // update gradient value
      _value = g;
      return _value;
    }    

  private:
    Add& _base;
  };

  _x.derivative(new Derivative_any(graph, *this));
  _y.derivative(new Derivative_any(graph, *this));
}

// F = x + y
const Tensor& Add::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get inputs
  auto& x = _x.forward();
  auto& y = _y.forward();

  // update value
  _value.noalias() = x + y;

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Sub
///////////////////////////////////////////

Sub::Sub(Graph& graph, Function& x, Function& y) : 
Function(graph), _x(x), _y(y)
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Sub& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = 1
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();

      // update gradient value
      _value = g;
      return _value;
    }    

  private:
    Sub& _base;
  };

  // Derivative with respect to y
  class Derivative_y : public Function
  {
  public:
    Derivative_y(Graph& graph, Sub& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdy = -1
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();

      // update gradient value
      _value = -g;
      return _value;
    }    

  private:
    Sub& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
  _y.derivative(new Derivative_y(graph, *this));
}

// F = x - y
const Tensor& Sub::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get inputs
  auto& x = _x.forward();
  auto& y = _y.forward();

  // update value
  _value.noalias() = x - y;

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Mul
///////////////////////////////////////////

Mul::Mul(Graph& graph, Function& x, Function& y) :
Function(graph), _x(x), _y(y)
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Mul& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = y
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& y = _base._y.forward();

      auto& dFdx = y;

      // update gradient value
      _value = g.array() * dFdx.array();

      return _value;
    }    

  private:
    Mul& _base;
  };

  // Derivative with respect to y
  class Derivative_y : public Function
  {
  public:
    Derivative_y(Graph& graph, Mul& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdy = x
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& x = _base._x.forward();

      auto& dFdy = x;

      // update gradient value
      _value = g.array() * dFdy.array();

      return _value;
    }    

  private:
    Mul& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
  _y.derivative(new Derivative_y(graph, *this));
}

// F = x * y
const Tensor& Mul::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get inputs
  auto& x = _x.forward();
  auto& y = _y.forward();

  // update value
  _value = x.array() * y.array();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Power
///////////////////////////////////////////

Power::Power(Graph& graph, Function& x, Function& y) :
Function(graph), _x(x), _y(y)
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Power& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = y * pow(F, y-1)
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& F = _base.forward();
      auto& g = _base.backward();
      auto& x = _base._x.forward();
      auto& y = _base._y.forward();

      auto dFdx = y.array() * x.array().pow(y.array() - 1);

      // update gradient value
      _value = g.array() * dFdx.array();
      return _value;
    }    

  private:
    Power& _base;
  };

  // Derivative with respect to y
  class Derivative_y : public Function
  {
  public:
    Derivative_y(Graph& graph, Power& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdy = F * log(x)
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& F = _base.forward();
      auto& g = _base.backward();
      auto& x = _base._x.forward();
      auto& y = _base._y.forward();

      auto dFdy = F.array() * x.array().log();

      // update gradient value
      _value = g.array() * dFdy.array();
      return _value;
    }    

  private:
    Power& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
  _y.derivative(new Derivative_y(graph, *this));
}

// F = pow(x, y)
const Tensor& Power::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get inputs
  auto& x = _x.forward();
  auto& y = _y.forward();

  // update value
  _value = x.array().pow(y.array());

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Tanh
///////////////////////////////////////////

Tanh::Tanh(Graph& graph, Function& x) : Function(graph), _x(x)
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Tanh& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = 1 - (tanh(x))^2
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& F = _base.forward();

      auto dFdx = 1 - F.array() * F.array();

      // update gradient value
      _value = g.array() * dFdx.array();

      return _value;
    }    

  private:
    Tanh& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
}

// F = (exp(x) - exp(-x) / (exp(x) + exp(-x))
const Tensor& Tanh::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _x.forward();

  // update value
  _value = x.array().tanh();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Sigmoid
///////////////////////////////////////////

Sigmoid::Sigmoid(Graph& graph, Function& x) :
Function(graph), _x(x)
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Sigmoid& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = F(1 - F)
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& F = _base.forward();

      auto dFdx = F.array() * (1 - F.array());

      // update gradient value
      _value = g.array() * dFdx.array();

      return _value;
    }    

  private:
    Sigmoid& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
}

// F = 1 / (1 + exp(-x))
const Tensor& Sigmoid::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _x.forward();

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
  auto& zero = scalar(graph, 0);

  // create relu from function max and constant '0'
  _relu = graph.new_max(x, *graph.new_broadcast(zero, x));

  // let output handle the gradient directly
  _relu->derivative(_graph.new_iderivative(*this));
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
Function(graph), _x(x), _rate(rate), _enabled(true)
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Dropout& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdy = x * mask
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& m = _base._mask;

      auto dFdx = g.array() * m.array();

      // update gradient value
      _value = dFdx;
      return _value;
    }

  private:
    Dropout& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
}

// F = x * mask
const Tensor& Dropout::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get values
  auto& x = _x.forward();

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
// Function Softmax
///////////////////////////////////////////

// Derivative approches 0 when error is evently distributed.
// Use LogSoftmax for training and Softmax for inference.
Softmax::Softmax(Graph& graph, Function& x) :
Function(graph), _x(x)
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Softmax& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx(j,i) = F(i)(K(i,j) - F(j))
    // K(i,j) = 1 for i==j, 0 for i!=j (Kronecker delta)
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& y = _base.forward();

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

  private:
    Softmax& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
}

// F = exp(x) / sum(exp(x))
const Tensor& Softmax::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input vector
  auto& x = _x.forward();

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
Function(graph), _x(x)
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Softplus& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = sigmoid(x) = 1 / (1 + exp(-x))
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      // get input
      auto& x = _base._x.forward();

      // update gradient
      //_value = 1 / (1 + (-x).array().exp());
      // for numerical stability compute sigmoid with tanh
      _value = 0.5 * (x.array() * 0.5).tanh() + 0.5;

      // return value
      return _value;
    }    

  private:
    Softplus& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
}

// F = log(1 + exp(x))
const Tensor& Softplus::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input vector
  auto& x = _x.forward();

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
Function(graph), _x(x)
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, LogSoftmax& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx(j,i) = K(i,j) - F(j)
    // K(i,j) = 1 for i==j, 0 for i!=j (Kronecker delta)
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto s = _base.softmax();

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

  private:
    LogSoftmax& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
}

// F = exp(x) / sum(exp(x))
const Tensor LogSoftmax::softmax()
{
  // get input vector
  auto& x = _x.forward();

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

  auto& x = _x.forward();

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
Function(graph), _x(x)
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Log& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = 1 / x
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& x = _base._x.forward();

      auto dFdx = 1 / x.array();

      // update gradient value
      _value = g.array() * dFdx.array();

      return _value;
    }    

  private:
    Log& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
}

// F = log(x)
const Tensor& Log::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _x.forward();

  // update value
  _value = x.array().log();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Abs
///////////////////////////////////////////

Abs::Abs(Graph& graph, Function& x) :
Function(graph), _x(x)
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Abs& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = abs(x) / x
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& x = _base._x.forward();

      // Taking sing() of input, gives the derivative values,
      // and for x == 0 the derivative becomes 0, which is half
      // of the derivative maximum and minimum value.
      auto dFdx = x.array().sign();

      // update gradient value
      _value = g.array() * dFdx.array();

      return _value;
    }    

  private:
    Abs& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
}

// F = abs(x)
const Tensor& Abs::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _x.forward();

  // update value
  _value = x.array().abs();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Transpose
///////////////////////////////////////////

Transpose::Transpose(Graph& graph, Function& x) :
Function(graph), _x(x)
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Transpose& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = base.dFdx.T
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();

      // update gradient value
      _value = g.transpose();

      // return gradient value
      return _value;
    }    

  private:
    Transpose& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
}

// F = x.T
const Tensor& Transpose::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _x.forward();

  // update value
  _value = x.transpose();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Sum
///////////////////////////////////////////

Sum::Sum(Graph& graph, Function& x) :
Function(graph), _x(x)
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Sum& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = 1
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& x = _base._x.forward();

      auto dFdx = g(0,0);

      // update gradient value
      _value = Tensor::Constant(x.rows(), x.cols(), dFdx);

     return _value;
    }    

  private:
    Sum& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
}

// F = sum(x)
const Tensor& Sum::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _x.forward();

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
Function(graph), _x(x)
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Mean& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = 1 / N
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& x = _base._x.forward();

      auto dFdx = g(0,0) / x.size();

      // update gradient value
      _value = Tensor::Constant(x.rows(), x.cols(), dFdx);

      return _value;
    }    

  private:
    Mean& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
}

// F = sum(x) / N
const Tensor& Mean::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _x.forward();

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
Function(graph), _x(x), _h(h)
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
  init();
}

GRU::GRU(Graph& graph, Function& x, Function& h, const GRU& other) :
Function(graph), _x(x), _h(h)
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
  init();
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

void GRU::init()
{
  // z
  auto& z = *_graph.new_sigmoid(
    product(_x,*_Wz) + product(_h,*_Uz) + *_bz);

  // r
  auto& r = *_graph.new_sigmoid(
    product(_x,*_Wr) + product(_h,*_Ur) + *_br);

  // c
  auto& c = *_graph.new_tanh(
    product(_x,*_Wh) + product(r * _h,*_Uh) + *_bh);

  // h(t)
  _GRU = &(z * _h + (1-z) * c);

  // let output handle the gradient directly
  _GRU->derivative(_graph.new_iderivative(*this));
}

///////////////////////////////////////////
// Function LSTM
///////////////////////////////////////////

LSTM::LSTM(Graph& graph, Function& x, Function& h, Function& c, int in, int out) :
Function(graph), _x(x), _h(h), _c(c)
{
  // construct new variables (row-major)
  _Wi = graph.new_variable(in, out, "Wi");
  _Hi = graph.new_variable(out, out, "Hi");
  _bi = graph.new_variable(1, out, "bi");

  _Wf = graph.new_variable(in, out, "Wf");
  _Hf = graph.new_variable(out, out, "Hf");
  _bf = graph.new_variable(1, out, "bf");

  _Wo = graph.new_variable(in, out, "Wo");
  _Ho = graph.new_variable(out, out, "Ho");
  _bo = graph.new_variable(1, out, "bo");

  _Wg = graph.new_variable(in, out, "Wg");
  _Hg = graph.new_variable(out, out, "Hg");
  _bg = graph.new_variable(1, out, "bg");

  // build graph
  init();
}

LSTM::LSTM(Graph& graph, Function& x, Function& h, Function& c, const LSTM& other) :
Function(graph), _x(x), _h(h), _c(c)
{
  // share variables with the "other"
  _Wi = other._Wi;
  _Hi = other._Hi;
  _bi = other._bi;

  _Wf = other._Wf;
  _Hf = other._Hf;
  _bf = other._bf;

  _Wo = other._Wo;
  _Ho = other._Ho;
  _bo = other._bo;

  _Wg = other._Wg;
  _Hg = other._Hg;
  _bg = other._bg;

  // build graph
  init();
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

// f(t) = Sigmoid(Wf * x(t) + Hf * h(t-1) + bf)
// i(t) = Sigmoid(Wi * x(t) + Hi * h(t-1) + bi)
// o(t) = Sigmoid(Wo * x(t) + Ho * h(t-1) + bo)
// g(t) = Tanh   (Wg * x(t) + Hg * h(t-1) + bo)
// c(t) = f(t) . c(t-1) + i(t) . g(t)
// h(t) = o(t) . Tanh(c(t))
void LSTM::init()
{
  // f
  auto& f = *_graph.new_sigmoid(
    product(_x,*_Wf) + product(_h,*_Hf) + *_bf);

  // i
  auto& i = *_graph.new_sigmoid(
    product(_x,*_Wi) + product(_h,*_Hi) + *_bi);

  // o
  auto& o = *_graph.new_sigmoid(
    product(_x,*_Wo) + product(_h,*_Ho) + *_bo);

  // g
  auto& g = *_graph.new_tanh(
    product(_x,*_Wg) + product(_h,*_Hg) + *_bg);

  // c
  auto& c = f * _c + i * g;

  // h
  auto& h = o * *_graph.new_tanh(c);

  // set references
  _cell = &c;
  _LSTM = &h;

  // let output handle the gradient directly
  _LSTM->derivative(_graph.new_iderivative(*this));
}

///////////////////////////////////////////
// Sampler
///////////////////////////////////////////

Sampler::Sampler(Graph& graph, Function& m, Function& s) : 
Function(graph), _m(m), _s(s), _enabled(true)
{
  // random sampling with parametrization trick for back-propagation
  // Auto-Encoding Variational Bayes by Diederik P. Kingma, Max Welling
  // https://arxiv.org/pdf/1312.6114.pdf

  _e = _graph.new_constant();

  _Z = &(_m + *_e * _s);

  _Z->derivative(_graph.new_iderivative(*this));
}

const Tensor& Sampler::forward()
{
  // return cached value
  if (_value.size()) return _value;

  int rows = _m().rows();
  int cols = _m().cols();

  // sample random constant
  if (_enabled)
  {
    auto random = [&]() { return _graph.random().normal_dec(0, 1); };
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
// Norm
///////////////////////////////////////////

Norm::Norm(Graph& graph, Function& x, int rows, int cols, DTYPE eps) :
Function(graph), _x(x), _epsilon(eps)
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

  init();
}

Norm::Norm(Graph& graph, Function& x, const Norm& other) :
Function(graph), _x(x), _epsilon(other._epsilon)
{
  _a = other._a;
  _b = other._b;

  init();
}

void Norm::init()
{
  _H = _graph.new_constant(1,1); // dimension placeholder

  auto& mean = *_graph.new_sum(_x) / *_H;
  auto& x_mean = _x - *_graph.new_broadcast(mean, _x);

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

  _N->derivative(_graph.new_iderivative(*this));
}

// F = a * (x - m) / s - b
const Tensor& Norm::forward()
{
  // return cached value
  if (_value.size()) return _value;

  auto& x = _x.forward();

  // update dimension placeholder
  _H->value() = Tensor::Constant(1,1, x.rows() * x.cols());

  // compute norm
  _value = _N->forward();

  return _value;
}

///////////////////////////////////////////
// Gaussian
///////////////////////////////////////////

Gaussian::Gaussian(Graph& graph, Function& x, Function& m, Function& s) : 
Function(graph)
{
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
    Derivative_a(Graph& graph, Gaussian& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFda = exp(z)
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& z = _base._z->forward();
      auto dFda = z.array().exp();

      // update gradient value
      _value = g.array() * dFda;

      // return gradient value
      return _value;
    }

  private:
    Gaussian& _base;
  };

  // Derivative with respect to z
  class Derivative_z : public Function
  {
  public:
    Derivative_z(Graph& graph, Function& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdz = a * exp(z) = F
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& dFdz = _base.forward();

      // update gradient value
      _value = g.array() * dFdz.array();

      // return gradient value
      return _value;
    }

  private:
    Function& _base;
  };

  _a->derivative(new Derivative_a(_graph, *this));
  _z->derivative(new Derivative_z(_graph, *this));
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
    Derivative_a(Graph& graph, LogGaussian& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFda = exp(z)
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& a = _base._a->forward();
      auto dFda = 1 / a.array();

      // update gradient value
      _value = g.array() * dFda;

      // return gradient value
      return _value;
    }

  private:
    LogGaussian& _base;
  };

  // Derivative with respect to z
  class Derivative_z : public Function
  {
  public:
    Derivative_z(Graph& graph, Function& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdz = a * exp(z) = F
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      // auto dFdz = Tensor::Ones(g.rows(), g.cols());

      // update gradient value
      _value = g.array() /** dFdz.array()*/;

      // return gradient value
      return _value;
    }

  private:
    Function& _base;
  };

  _a->derivative(new Derivative_a(_graph, *this));
  _z->derivative(new Derivative_z(_graph, *this));
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

Embedding::Embedding(Graph& graph, Function& i, int in, int out) :
Function(graph), _i(i)
{
  // construct new variables with row-wise embedding vectors
  _E = graph.new_variable(in, out, "Embedding.E");

  init();
}

Embedding::Embedding(Graph& graph, Function& i, const Embedding& other) :
Function(graph), _i(i)
{
  // share variables with the "other"
  _E = other._E;

  init();
}

void Embedding::init()
{
  // Derivative with respect to E
  class Derivative_E : public Function
  {
  public:
    Derivative_E(Graph& graph, Embedding& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdE = x(i)
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& E = _base._E->forward();

      // update gradient value
      _value = Tensor::Zero(E.rows(), E.cols());
      auto& index = _base._i();

      for (int i=0; i<index.size(); i++)
      {
        _value.row((int)index(i)) = g.row(i);
      }

      return _value;
    }

  private:
    Embedding& _base;
  };

  _E->derivative(new Derivative_E(_graph, *this));
}

// F = E * x(i)
const Tensor& Embedding::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get variables
  auto& E = _E->forward();

  // create initial value
  auto& index = _i();
  _value = Tensor::Zero(index.size(), E.cols());

  for (int i=0; i<index.size(); i++)
  {
    _value.row(i) = E.row((int)index(i));
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
_x(x),
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

  init();
}

Conv2D::Conv2D(Graph& graph, Function& x, const Conv2D& other) :
Function(graph), _x(x)
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

  init();
}

const Tensor& Conv2D::forward()
{
  // return cached value
  if (_value.size()) return _value;

  auto& x = _x();
  auto& K = K_matrix();

  //_value = K * x; // col major
  _value = ABT(x,K); // row major

  // return value
  return _value;
}

void Conv2D::init()
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Conv2D& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = K_matrix
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& K = _base.K_matrix();

      auto& dFdx = K;

      // update gradient value
      //_value = ABT(g, dFdx); // col major
      _value = g * dFdx; // row major

      return _value;
    }    

  private:
    Conv2D& _base;
  };

  // Derivative with respect to K
  class Derivative_K : public Function
  {
  public:
    Derivative_K(Graph& graph, Conv2D& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdK = x
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& x = _base._x.forward();
      auto& K = _base.K_matrix();

      auto& dFdK = x;
      //auto dFdK_matrix = ABT(g, dFdK, K); // col major
      auto dFdK_matrix = ATB(g, dFdK, K); // row major

      // update gradient value
      _value = _base.K_gradient(dFdK_matrix);

      return _value;
    }

  private:
    Conv2D& _base;
  };

  _x.derivative(new Derivative_x(_graph, *this));
  _K->derivative(new Derivative_K(_graph, *this));
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
// Function Error
///////////////////////////////////////////

Erf::Erf(Graph& graph, Function& x) :
Function(graph), _x(x)
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Erf& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = (2 / sqrt(pi)) * exp(-x^2)
    virtual const Tensor& forward()
    {
      // return cached gradient
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& x = _base._x.forward();

      // M_2_SQRTPI = 2/sqrt(pi)
      auto dFdx = M_2_SQRTPI * (-x.array() * x.array()).exp();

      // update gradient value
      _value = dFdx.array() * g.array();

     return _value;
    }

  private:
    Erf& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
}

// F = erf(x)
const Tensor& Erf::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _x.forward();

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
  // M_2_PI = 2 / pi
  //_gelu = &(0.5 * x * (1.0 + *graph.new_tanh(
  //  M_2_PI * (x + 0.044715 * *graph.new_power(x, 3))
  //)));

  _gelu = &(x * *graph.new_sigmoid(1.702 * x));

  _gelu->derivative(graph.new_iderivative(*this));
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
Function(graph), _x(x)
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, GeLU& base) :
    Function(graph), _base(base) { graph.keep(this); }

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

      auto& g = _base.backward();
      auto& x = _base._x.forward();

      auto z = x.array() * M_SQRT1_2;
      auto derfdz = M_2_SQRTPI * (-z.array() * z.array()).exp();
      Tensor dFdx = 0.5 * (1 + z.erf()) + 0.5 * x.array() * derfdz * M_SQRT1_2;

      // update gradient value
      _value = g.array() * dFdx.array();

      return _value;
    }

  private:
    GeLU& _base;
  };

  _x.derivative(new Derivative_x(graph, *this));
}

// F = 0.5 * x * (1 + erf(x / sqrt(2)))
const Tensor& GeLU::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input
  auto& x = _x.forward();

  // update value
  _value = 0.5 * x.array() * (1 + (x.array() * M_SQRT1_2).erf());

  // return value
  return _value;
}

#endif

///////////////////////////////////////////
// Graph
///////////////////////////////////////////

void Graph::clear()
{
  for (auto e: _nodes) delete e;
  _nodes.clear();
  _vars.clear();
  _names.clear();
}

// set function name
Function* Graph::name(Function* f, const char* name)
{
  if (f == nullptr || name == nullptr)
    return nullptr;

  auto it = std::find(_nodes.begin(), _nodes.end(), f);
  if (it != _nodes.end())
  {
    _names[it - _nodes.begin()] = name;
    return *it;
  }
  else
  {
    return nullptr;
  }
}

// get function by name
Function* Graph::function(const char* name) const
{
  if (name == nullptr)
    return nullptr;

  auto it = std::find(_names.begin(), _names.end(), name);
  if (it != _names.end())
  {
    return _nodes[it - _names.begin()];
  }
  else
  {
    return nullptr;
  }
}

// get graph variables with unique names
std::map<std::string, Variable*> Graph::named_variables() const
{
  // get variable positions in node list
  std::vector<uint32_t> inode;
  for (int i=0; i<_nodes.size() && inode.size() != _vars.size(); i++)
  {
    if (_nodes[i] == _vars[inode.size()])
    {
      inode.push_back(i);
    }
  }

  // keep track of name counts
  std::map<std::string, int> counts;
  std::map<std::string, Variable*> dict;

  for (int i=0; i<_vars.size(); i++)
  {
    // get variable name
    auto name = _names[inode[i]];

    // index duplicate names
    auto itc = counts.find(name);
    if (itc == counts.end())
    {
      counts[name] = 0;
    }
    else
    {
      counts[name] = itc->second + 1;
      name = name + "." + std::to_string(counts[name]);
    }

    // add name:variable to dictionary
    dict[name] = _vars[i];
  }

  return dict;
}

// track function
void Graph::keep(Function* f, const char* name)
{
  _nodes.push_back(f);
  _names.push_back((name) ? name : "");
}

// track variable
void Graph::keep(Variable* v, const char* name)
{
  _nodes.push_back(v);
  _vars.push_back(v);
  _names.push_back((name) ? scope_name() + name : scope_name() + "Variable");
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

