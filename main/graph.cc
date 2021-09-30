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

#include "graph.hh"

namespace seegnify {

///////////////////////////////////////////
// Function operators
///////////////////////////////////////////

Function& power(Function& x, DTYPE y)
{
  auto& c = *x.graph().new_scalar(y, x);
  return *x.graph().new_power(x, c);
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
  auto& c = *x.graph().new_scalar(-1, x);
  return *x.graph().new_mul(c, x);
}

// DTYPE operators
Function& operator+(DTYPE x, Function& y)
{
  auto& c = *y.graph().new_scalar(x, y);
  return *y.graph().new_add(c, y);
}

Function& operator+(Function& x, DTYPE y)
{
  auto& c = *x.graph().new_scalar(y, x);
  return *x.graph().new_add(x, c);
}

Function& operator-(DTYPE x, Function& y)
{
  auto& c = *y.graph().new_scalar(x, y);
  return *y.graph().new_sub(c, y);
}

Function& operator-(Function& x, DTYPE y)
{
  auto& c = *x.graph().new_scalar(y, x);
  return *x.graph().new_sub(x, c);
}

Function& operator*(DTYPE x, Function& y)
{
  auto& c = *y.graph().new_scalar(x, y);
  return *y.graph().new_mul(c, y);
}

Function& operator*(Function& x, DTYPE y)
{
  auto& c = *x.graph().new_scalar(y, x);
  return *x.graph().new_mul(x, c);
}

Function& operator/(Function& x, DTYPE y)
{
  return x * (1.0/y);
}

Function& operator/(DTYPE x, Function& y)
{
  auto& c = *y.graph().new_scalar(x, y);
  return c * power(y, -1);
}

Function& operator/(Function& x, Function& y)
{
  return x * power(y, -1);
}

///////////////////////////////////////////
// Identity Derivarive impl
///////////////////////////////////////////

// equivalent of "static class"
namespace
{
  // Identity (pass-through) derivative
  class IDerivative : public Function
  {
  public:
    IDerivative(Graph& graph, Function& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = base.dFdx
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();

      // update gradient value
      _value = g;

      // return gradient value
      return _value;
    }    

  private:
    Function& _base;
  };
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

static Tensor ABT(const Tensor& A, const Tensor& B)
{
  if (B.cols() == 1)
    return A * ConstRowVectorMap(B.data(), B.size());
  else
    return A * B.transpose();
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

    auto& aggregator = _graph.aggregator();

    if (_backprop) aggregator.aggregate(_gradient, _derivative);
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
// Function Scalar
///////////////////////////////////////////

Scalar::Scalar(Graph& graph, DTYPE scalar, Function& target) :
Function(graph), _target(target)
{
  _value.resize(1,1);
  _value << scalar;
}

// F = x (x broadcast to target)
const Tensor& Scalar::forward()
{ 
  auto& t = _target();
  if (_value.rows() != t.rows() || _value.cols() != t.cols())
  {
    _value = Tensor::Constant(t.rows(), t.cols(), _value(0,0));
  }
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

  auto& aggregator = _graph.aggregator();

  if (_backprop) aggregator.aggregate(_gradient, _derivative);

  return _gradient;
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
  // Split Derivative with respect toX
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

  // Split Derivative with respect toY
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
  ConstVectorMap vx(x.data(), x.size());
  ConstVectorMap vy(y.data(), y.size());
  Vector xy(x.size() + y.size());
  xy.topRows(x.size()) = vx;
  xy.bottomRows(y.size()) = vy;

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
      auto one = Tensor::Constant(x.rows(), x.cols(), 1);
      auto zero = Tensor::Constant(x.rows(), x.cols(), 0);
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
      auto one = Tensor::Constant(x.rows(), x.cols(), 1);
      auto zero = Tensor::Constant(x.rows(), x.cols(), 0);
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
      auto one = Tensor::Constant(x.rows(), x.cols(), 1);
      auto zero = Tensor::Constant(x.rows(), x.cols(), 0);
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
      auto one = Tensor::Constant(x.rows(), x.cols(), 1);
      auto zero = Tensor::Constant(x.rows(), x.cols(), 0);
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

Linear::Linear(Graph& graph, Function&x, int in, int out) : 
Function(graph), _x(x)
{
  // construct new variables
  _W = graph.new_variable(out, in);
  _b = graph.new_variable(out, 1);

  init();
}

Linear::Linear(Graph& graph, Function& x, const Linear& other) : 
Function(graph), _x(x)
{
  // share variables with the "other"
  _W = other._W;
  _b = other._b;

  init();
}

void Linear::init()
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Linear& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdx = W
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& W = _base._W->forward();

      auto& dFdx = W;

      // update gradient value
      _value = ATB(dFdx, g);
      return _value;
    }    

  private:
    Linear& _base;
  };

  // Derivative with respect to W
  class Derivative_W : public Function
  {
  public:
    Derivative_W(Graph& graph, Linear& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdW = x
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();
      auto& x = _base._x.forward();

      auto& dFdW = x;

      // update gradient value
      _value = ABT(g, dFdW);
      return _value;
    }    

  private:
    Linear& _base;
  };

  // Derivative with respect to b
  class Derivative_b : public Function
  {
  public:
    Derivative_b(Graph& graph, Linear& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdb = 1
    virtual const Tensor& forward()
    {
      // return cached value
      if (_value.size()) return _value;

      auto& g = _base.backward();

      // update value
      _value = g;
      return _value;
    }    

  private:
    Linear& _base;
  };

  _x.derivative(new Derivative_x(_graph, *this));
  _W->derivative(new Derivative_W(_graph, *this));
  _b->derivative(new Derivative_b(_graph, *this));
}

// F = W * x + b
const Tensor& Linear::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // get input vector
  auto& x = _x.forward();

  // get variables
  auto& W = _W->forward();
  auto& b = _b->forward();

  // create cached value
  _value.noalias() = W * x + b;

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
  // create relu from function max and constant '0'
  _relu = graph.new_max(x, *graph.new_scalar(0, x));

  // let output handle the gradient directly
  _relu->derivative(new IDerivative(_graph, *this));
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
// Function Step
///////////////////////////////////////////

Step::Step(Graph& graph, Function& x, DTYPE lo, DTYPE hi) :
Function(graph), _x(x), _lo(lo), _hi(hi)
{
  // assume 'on purpose' that the derivarive of 'step' is 1
  x.derivative(new IDerivative(_graph, *this));
}

// F = {lo for x <= 0, hi for x > 0}
const Tensor& Step::forward()
{
  // return cached value
  if (_value.size()) return _value;

  auto& x = _x.forward();

  // update value
  auto lo = Tensor::Constant(x.rows(), x.cols(), _lo);
  auto hi = Tensor::Constant(x.rows(), x.cols(), _hi);
  _value = (x.array() <= 0).select(lo, hi);

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Dropout
///////////////////////////////////////////

Dropout::Dropout(Graph& graph, Function& x, DTYPE rate) :
Function(graph), _x(x), _rate(rate)
{
  // Derivative with respect to x
  class Derivative_x : public Function
  {
  public:
    Derivative_x(Graph& graph, Dropout& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdy = x.T
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
  auto random = [&]() { return _graph.random().uniform_dec(0, 1); };
  auto mask = Tensor::NullaryExpr(x.rows(), x.cols(), random);
  auto zero = Tensor::Constant(x.rows(), x.cols(), 0);
  auto ones = Tensor::Constant(x.rows(), x.cols(), 1);
  _mask = (mask.array() < _rate).select(zero, ones);

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
      auto& F = _base.forward();

      // use Identity matrix to construct diagonal matrix
      auto I = Tensor::Identity(F.rows(), F.rows());
      auto dFdx = I * F.asDiagonal() - F * F.transpose();

      // multiply each row of dFdx by coresponding coefficient from d
      _value = (g.asDiagonal() * dFdx).colwise().sum().transpose();

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

  // for numerical stability subtract max(_value) before exponential
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
  auto zero = Tensor::Constant(x.rows(), x.cols(), 0);
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
      auto S = _base.softmax();

      auto I = Tensor::Identity(S.rows(), S.rows());
      auto dFdx = I - Tensor::Ones(S.rows(), 1) * S.transpose();

      // multiply each row of dFdx by coresponding coefficient from g
      _value = (g.asDiagonal() * dFdx).colwise().sum().transpose();

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

  // for numerical stability subtract max(_value) before exponential
  auto exp_x = (x.array() - x.maxCoeff()).exp();

  // calculate softmax
  return exp_x / exp_x.sum();
}

// F = log(exp(x) / sum(exp(x)))
const Tensor& LogSoftmax::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // create cached value
  _value = softmax().array().log();

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
  // construct new variables
  _Wz = graph.new_variable(out, in);
  _Uz = graph.new_variable(out, out);
  _bz = graph.new_variable(out, 1);

  _Wr = graph.new_variable(out, in);
  _Ur = graph.new_variable(out, out);
  _br = graph.new_variable(out, 1);

  _Wh = graph.new_variable(out, in);
  _Uh = graph.new_variable(out, out);
  _bh = graph.new_variable(out, 1);

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
    product(*_Wz,_x) + product(*_Uz,_h) + *_bz);

  // r
  auto& r = *_graph.new_sigmoid(
    product(*_Wr,_x) + product(*_Ur,_h) + *_br);

  // c
  auto& c = *_graph.new_tanh(
    product(*_Wh,_x) + product(*_Uh,r * _h) + *_bh);

  // h(t)
  _GRU = &(z * _h + (1-z) * c);

  // let output handle the gradient directly
  _GRU->derivative(new IDerivative(_graph, *this));
}

///////////////////////////////////////////
// Function AGRU
///////////////////////////////////////////

AGRU::AGRU(Graph& graph, Function& x, Function& h, int in, int out) :
Function(graph), _x(x), _h(h)
{
  // construct new variables
  _Wz = graph.new_variable(out, in);
  _Uz = graph.new_variable(out, out);
  _bz = graph.new_variable(out, 1);

  _Wr = graph.new_variable(out, in);
  _Ur = graph.new_variable(out, out);
  _br = graph.new_variable(out, 1);

  _Wp = graph.new_variable(in, in);
  _Up = graph.new_variable(in, out);
  _bp = graph.new_variable(in, 1);

  _Wq = graph.new_variable(out, in);
  _Uq = graph.new_variable(out, out);
  _bq = graph.new_variable(out, 1);

  _Wh = graph.new_variable(out, in);
  _Uh = graph.new_variable(out, out);
  _bh = graph.new_variable(out, 1);

  // build graph
  init();
}

AGRU::AGRU(Graph& graph, Function& x, Function& h, const AGRU& other) :
Function(graph), _x(x), _h(h)
{
  // share variables with the "other"
  _Wz = other._Wz;
  _Uz = other._Uz;
  _bz = other._bz;

  _Wr = other._Wr;
  _Ur = other._Ur;
  _br = other._br;

  _Wp = other._Wp;
  _Up = other._Up;
  _bp = other._bp;

  _Wq = other._Wq;
  _Uq = other._Uq;
  _bq = other._bq;

  _Wh = other._Wh;
  _Uh = other._Uh;
  _bh = other._bh;

  // build graph
  init();
}

const Tensor& AGRU::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // create value h(t)
  _value = _AGRU->forward();

  // return value
  return _value;
}

void AGRU::init()
{
  // z
  auto& z = *_graph.new_sigmoid(
    product(*_Wz,_x) + product(*_Uz,_h) + *_bz);

  // r
  auto& r = *_graph.new_sigmoid(
    product(*_Wr,_x) + product(*_Ur,_h) + *_br);

  // p
  auto& p = *_graph.new_sigmoid(
    product(*_Wp,_x) + product(*_Up,_h) + *_bp);

  // q
  auto& q = *_graph.new_sigmoid(
    product(*_Wq,_x) + product(*_Uq,_h) + *_bq);

  // c
  auto& c = *_graph.new_tanh(
    product(*_Wh,p * _x) + product(*_Uh,r * _h) + q * *_bh);

  // h
  _AGRU = &(z * _h + (1-z) * c);

  // let output handle the gradient directly
  _AGRU->derivative(new IDerivative(_graph, *this));
}

///////////////////////////////////////////
// Function LSTM
///////////////////////////////////////////

LSTM::LSTM(Graph& graph, Function& x, Function& h, Function& c, int in, int out) :
Function(graph), _x(x), _h(h), _c(c)
{
  // construct new variables
  _Wi = graph.new_variable(out, in);
  _Hi = graph.new_variable(out, out);
  _Ci = graph.new_variable(out, out);
  _bi = graph.new_variable(out, 1);

  _Wf = graph.new_variable(out, in);
  _Hf = graph.new_variable(out, out);
  _Cf = graph.new_variable(out, out);
  _bf = graph.new_variable(out, 1);

  _Wo = graph.new_variable(out, in);
  _Ho = graph.new_variable(out, out);
  _Co = graph.new_variable(out, out);
  _bo = graph.new_variable(out, 1);

  _Wc = graph.new_variable(out, in);
  _Hc = graph.new_variable(out, out);
  _bc = graph.new_variable(out, 1);

  // build graph
  init();
}

LSTM::LSTM(Graph& graph, Function& x, Function& h, Function& c, const LSTM& other) :
Function(graph), _x(x), _h(h), _c(c)
{
  // share variables with the "other"
  _Wi = other._Wi;
  _Hi = other._Hi;
  _Ci = other._Ci;
  _bi = other._bi;

  _Wf = other._Wf;
  _Hf = other._Hf;
  _Cf = other._Cf;
  _bf = other._bf;

  _Wo = other._Wo;
  _Ho = other._Ho;
  _Co = other._Co;
  _bo = other._bo;

  _Wc = other._Wc;
  _Hc = other._Hc;
  _bc = other._bc;

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

// f(t) = Sigmoid(Wf * x(t) + Hf * h(t-1) + Cf * c(t-1) + bf)
// i(t) = Sigmoid(Wi * x(t) + Hi * h(t-1) + Ci * c(t-1) + bi)
// c(t) = f(t) . c(t-1) + i(t) . Tanh(Wc * x(t) + Hc * h(t-1) + bc)
// o(t) = Sigmoid(Wo * x(t) + Ho * h(t-1) + Co * c(t) + bo)
// h(t) = o(t) . Tanh(c(t))
void LSTM::init()
{
  // f
  auto& f = *_graph.new_sigmoid(
    product(*_Wf,_x) + product(*_Hf,_h) + product(*_Cf,_c) + *_bf);

  // i
  auto& i = *_graph.new_sigmoid(
    product(*_Wi,_x) + product(*_Hi,_h) + product(*_Ci,_c) + *_bi);

  // c
  auto& c = f * _c + i * *_graph.new_tanh(
    product(*_Wc,_x) + product(*_Hc,_h) + *_bc);

  // o
  auto& o = *_graph.new_sigmoid(
    product(*_Wo,_x) + product(*_Ho,_h) + product(*_Co,c) + *_bo);

  // h
  auto& h = o * *_graph.new_tanh(c);

  // set references
  _cell = &c;
  _LSTM = &h;

  // let output handle the gradient directly
  _LSTM->derivative(new IDerivative(_graph, *this));
}

///////////////////////////////////////////
// Function FGU
///////////////////////////////////////////

FGU::FGU(Graph& graph, Function& x, int in, int out) : 
Function(graph), _x(x)
{
  // construct new variables
  _Lp = graph.new_linear(x, in, in);
  _Lq = graph.new_linear(x, in, out);

  _Wh = graph.new_variable(out, in);
  _bh = graph.new_variable(out, 1);

  // build graph
  init();
}

FGU::FGU(Graph& graph, Function& x, const FGU& other) :
Function(graph), _x(x)
{
  // share variables with the "other"
  _Lp = graph.new_linear(x, *other._Lp);
  _Lq = graph.new_linear(x, *other._Lq);

  _Wh = other._Wh;
  _bh = other._bh;

  // build graph
  init();
}

const Tensor& FGU::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // create value h(t)
  _value = _FGU->forward();

  // return value
  return _value;
}

void FGU::init()
{
  // p
  auto& p = *_graph.new_sigmoid(*_Lp);

  // q
  auto& q = *_graph.new_sigmoid(*_Lq);

  // h
  _FGU = &(q * *_graph.new_tanh(product(*_Wh,p * _x) + *_bh));

  // let output handle the gradient directly
  _FGU->derivative(new IDerivative(_graph, *this));
}

///////////////////////////////////////////
// Sampler
///////////////////////////////////////////

Sampler::Sampler(Graph& graph, Function& m, Function& s) : 
Function(graph), _m(m), _s(s)
{
  // random sampling is not differentiatable,
  // therefore no derivatives are created for m or s.
}

const Tensor& Sampler::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // initialize value
  _value.resize(_m().rows(), _m().cols());

  auto m = _m().data();
  auto s = _s().data();
  auto v = _value.data();
  auto& rng = _graph.random();

  // sample value
  auto size = std::min(_m().size(), _s().size());
  for (auto i=0; i<size; i++) { v[i] = rng.normal_dec(m[i], s[i]); }

  // return value
  return _value;
}

///////////////////////////////////////////
// Gaussian
///////////////////////////////////////////

Gaussian::Gaussian(Graph& graph, Function& x, Function& m, Function& s) : 
Function(graph)
{
  auto& x_m = *graph.new_sub(x, m);
  auto& x_m_2 = *graph.new_mul(x_m, x_m);
  auto& s_2 = *graph.new_mul(s, s);

  // compute argument for exp(z)
  _z = &(-0.5 * x_m_2 / s_2);

  // for performance and numerical stability we do NOT add exp(z) to the graph,
  // instead, we calculate exp(z) in forward() method and use
  // F as gradient of exp(z) in the z derivative:

  // Derivative with respect to z
  class Derivative_z : public Function
  {
  public:
    Derivative_z(Graph& graph, Function& base) :
    Function(graph), _base(base) { graph.keep(this); }

    // dFdz = exp(z) = F
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

  _z->derivative(new Derivative_z(_graph, *this));
}

// Z = -(x - m)^2 / (2*s^2)
// F = exp(Z)
const Tensor& Gaussian::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // F = exp(Z)
  _value = _z->forward().array().exp();

  // return value
  return _value;
}

///////////////////////////////////////////
// Log-Guassian
///////////////////////////////////////////

LogGaussian::LogGaussian(Graph& graph, Function& x, Function& m, Function& s) : 
Function(graph)
{
  auto& x_m = *graph.new_sub(x, m);
  auto& x_m_2 = *graph.new_mul(x_m, x_m);
  auto& s_2 = *graph.new_mul(s, s);

  // compute z for log(exp(z))
  _z = &(-0.5 * x_m_2 / s_2);

  // let output handle the gradient directly
  _z->derivative(new IDerivative(_graph, *this));  
}

// Z = -(x - m)^2 / (2*s^2)
// F = log(exp(Z)) = Z
const Tensor& LogGaussian::forward()
{
  // return cached value
  if (_value.size()) return _value;

  _value = _z->forward();

  // return value
  return _value;
}

///////////////////////////////////////////
// Function Hopfield
///////////////////////////////////////////

Hopfield::Hopfield(Graph& graph, Function& x, DTYPE b, int size, int count) :
Function(graph)
{
  // construct new variables
  _W = graph.new_variable(size, count);
  _b = b;

  init(x);
}

Hopfield::Hopfield(Graph& graph, Function& x, const Hopfield& other) :
Function(graph)
{
  // share variables with the "other"
  _W = other._W;
  _b = other._b;

  init(x);
}

void Hopfield::init(Function& x)
{
  // W.T
  auto& WT = *_graph.new_transpose(*_W);

  // WTx
  auto& WTx = *_graph.new_product(WT, x);

  // softmax
  auto& softmax = *_graph.new_softmax(_b * WTx);

  // H
  _H = _graph.new_product(*_W, softmax);

  // let output handle the gradient directly
  _H->derivative(new IDerivative(_graph, *this));
}

// F = W * softmax(b * W.T * x)
const Tensor& Hopfield::forward()
{
  // return cached value
  if (_value.size()) return _value;

  // update value
  _value = _H->forward();

  // return value
  return _value;
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

// get function by id
Function* Graph::function(int id) const
{
  if (id < 0 or id >= _nodes.size())
    return nullptr;
  else
    return _nodes[id];
}

// get function by name
Function* Graph::function(const char* name) const
{
  if (name == nullptr)
    return nullptr;

  auto it = _names.find(name);
  if (it == _names.end())
    return nullptr;
  else
    return it->second;
}

// track function
void Graph::keep(Function* f, const char* name)
{
  _nodes.push_back(f);
  if (name != nullptr)
    _names[name] = f;
}

// track variable
void Graph::keep(Variable* v, const char* name)
{
  _nodes.push_back(v);
  _vars.push_back(v);
  if (name != nullptr)
    _names[name] = v;
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
  int y_rows = f.forward().rows();
  int y_cols = f.forward().cols();
  int x_rows = x.forward().rows();
  int x_cols = x.forward().cols();
  Tensor dfdx = Tensor::Zero(x_rows, x_cols);
  
  for (int yr=0; yr<y_rows; yr++)
  for (int yc=0; yc<y_cols; yc++)
  for (int xr=0; xr<x_rows; xr++)
  for (int xc=0; xc<x_cols; xc++)
  dfdx(xr, xc) = dfdx(xr, xc) + dFdX(f, x, yr, yc, xr, xc);

  return dfdx;
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

