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

#ifndef _SEEGNIFY_GRAPH_H_
#define _SEEGNIFY_GRAPH_H_

#include <memory>
#include <iostream>
#include <unordered_map>

#include "types.hh"
#include "random.hh"

namespace seegnify {

// Base class declarations
class Aggregator;
class Function;
class Graph;

// Function operators
Function& operator+(Function& x, Function& y);
Function& operator-(Function& x, Function& y);
Function& operator*(Function& x, Function& y);
Function& operator-(Function& x);
Function& operator+(DTYPE x, Function& y);
Function& operator+(Function& x, DTYPE y);
Function& operator-(DTYPE x, Function& y);
Function& operator-(Function& x, DTYPE y);
Function& operator*(DTYPE x, Function& y);
Function& operator*(Function& x, DTYPE y);
Function& operator/(Function& x, DTYPE y);
Function& operator/(DTYPE x, Function& y);
Function& operator/(Function& x, Function& y);

// Differentiable Function class
class Function
{
public:
  // default ctor
  Function(Graph& graph);

  // disable copy ctor
  Function(const Function& f) = delete;

  // forward traversal
  virtual const Tensor& forward() = 0;

  // backward traversal
  virtual const Tensor& backward();

  // recache function value and gradient
  virtual void recache();

  // set derivative callback
  void derivative(Function* d) { _derivative.push_back(d); }

  // enable / disable back-prop
  void backprop(bool enable) { _backprop = enable; }

  // variable value
  Tensor& value() { return _value; }

  // gradient value
  Tensor& gradient() { return _gradient; }

  // value operator
  const Tensor& operator()() { return forward(); }

  // graph access
  Graph& graph() { return _graph; }

protected:
  // backprop flag
  bool _backprop;

  // derivative callbacks
  std::vector<Function*> _derivative;

  // function value cache
  Tensor _value;

  // function gradient cache
  Tensor _gradient;

  // function graph
  Graph& _graph;
};

// Constant function
class Constant : public Function
{
public:
  Constant(Graph& graph, int rows = 0, int cols = 0);

  virtual const Tensor& forward() { return _value; }

  virtual void recache() { /* no-op */ }
};

// Variable function
class Variable : public Function
{
public:
  Variable(Graph& graph, int rows = 0, int cols = 0);

  virtual const Tensor& forward() { return _value; }

  virtual const Tensor& backward();

  virtual void recache() { /* no-op */ }
};

// Broadcast function
class Broadcast : public Function
{
public:
  Broadcast(Graph& graph, Function& x, Function& target);

  virtual const Tensor& forward();

protected:
  Function& _x;
  Function& _t;
};

// Split function (takes a fragment of the input)
class Split : public Function
{
public:
  Split(Graph& graph, Function& x, int r, int c, int rows, int cols);

  virtual const Tensor& forward();

protected:
  const int _r;
  const int _c;
  const int _rows;
  const int _cols;
  Function& _x;
};

// Join function (takes two inputs and creates concatenated output)
class Join : public Function
{
public:
  Join(Graph& graph, Function& x, Function& y, int rows, int cols);

  virtual const Tensor& forward();

protected:
  const int _rows;
  const int _cols;
  Function& _x;
  Function& _y;
};

// Min function (takes two inputs and creates min output)
class Min : public Function
{
public:
  Min(Graph& graph, Function& x, Function& y);

  virtual const Tensor& forward();

protected:
  Function& _x;
  Function& _y;
};

// Max function (takes two inputs and creates max output)
class Max : public Function
{
public:
  Max(Graph& graph, Function& x, Function& y);

  virtual const Tensor& forward();

protected:
  Function& _x;
  Function& _y;
};

// Linear function
class Linear : public Function
{
public:
  Linear(Graph& graph, Function& x, int in, int out);
  Linear(Graph& graph, Function& x, const Linear& other);

  // variable access
  Variable& W() { return *_W; }
  Variable& b() { return *_b; }

  virtual const Tensor& forward();

private:
  void init();

protected:
  Function& _x;
  Variable* _W;
  Variable* _b;
};

// Product (linear unbiased) function
class Product : public Function
{
public:
  Product(Graph& graph, Function& x, Function& y);

  virtual const Tensor& forward();

protected:
  Function& _x;
  Function& _y;
};

// Add function
class Add : public Function
{
public:
  Add(Graph& graph, Function& x, Function& y);

  virtual const Tensor& forward();

protected:
  Function& _x;
  Function& _y;
};

// Subtract function
class Sub : public Function
{
public:
  Sub(Graph& graph, Function& x, Function& y);

  virtual const Tensor& forward();

protected:
  Function& _x;
  Function& _y;
};

// Multiply function
class Mul : public Function
{
public:
  Mul(Graph& graph, Function& x, Function& y);

  virtual const Tensor& forward();

protected:
  Function& _x;
  Function& _y;
};

// Power function
class Power : public Function
{
public:
  Power(Graph& graph, Function& x, Function& y);

  virtual const Tensor& forward();

protected:
  Function& _x;
  Function& _y;
};

// Tanh function
class Tanh : public Function
{
public:
  Tanh(Graph& graph, Function& x);

  virtual const Tensor& forward();

protected:
  Function& _x;
};

// Sigmoid function
class Sigmoid : public Function
{
public:
  Sigmoid(Graph& graph, Function& x);

  virtual const Tensor& forward();

protected:
  Function& _x;
};

// ReLU function
class ReLU : public Function
{
public:
  ReLU(Graph& graph, Function& x);

  virtual const Tensor& forward();

protected:
  Function* _relu;
};

// Step function
class Step : public Function
{
public:
  Step(Graph& graph, Function& x, DTYPE lo = 0, DTYPE hi = 1);

  virtual const Tensor& forward();

protected:
  DTYPE _lo;
  DTYPE _hi;
  Function& _x;
};

// Dropout function
class Dropout : public Function
{
public:
  Dropout(Graph& graph, Function& x, DTYPE rate);

  virtual const Tensor& forward();

protected:
  DTYPE _rate;
  Tensor _mask;
  Function& _x;
};

// Softmax function
class Softmax : public Function
{
public:
  Softmax(Graph& graph, Function& x);

  virtual const Tensor& forward();

protected:
  Function& _x;
};

// Softplus function
class Softplus : public Function
{
public:
  Softplus(Graph& graph, Function& x);

  virtual const Tensor& forward();

protected:
  Function& _x;
};

// Log-Softmax function
class LogSoftmax : public Function
{
public:
  LogSoftmax(Graph& graph, Function& x);

  const Tensor softmax();
  virtual const Tensor& forward();

protected:
  Function& _x;
};

// Logarithm function
class Log : public Function
{
public:
  Log(Graph& graph, Function& x);

  virtual const Tensor& forward();

protected:
  Function& _x;
};

// Absolute function
class Abs : public Function
{
public:
  Abs(Graph& graph, Function& x);

  virtual const Tensor& forward();

protected:
  Function& _x;
};

// Transpose function
class Transpose : public Function
{
public:
  Transpose(Graph& graph, Function& x);

  virtual const Tensor& forward();

protected:
  Function& _x;
};

// Sum reduction function
class Sum : public Function
{
public:
  Sum(Graph& graph, Function& x);

  virtual const Tensor& forward();

protected:
  Function& _x;
};

// Mean reduction function
class Mean : public Function
{
public:
  Mean(Graph& graph, Function& x);

  virtual const Tensor& forward();

protected:
  Function& _x;
};

// Standard GRU function
// z(t) = Sigmoid(Wz * x(t) + Uz * h(t-1) + bz)
// r(t) = Sigmoid(Wr * x(t) + Ur * h(t-1) + br)
// c(t) = Tanh(Wh * x(t) + Uh * (r(t) . h(t-1)) + bh)
// h(t) = z(t) . h(t-1) + (1-z(t)) . c(t)
class GRU : public Function
{
public:
  GRU(Graph& graph, Function& x, Function& h, int in, int out);
  GRU(Graph& graph, Function& x, Function& h, const GRU& other);

  Variable& Wz() { return *_Wz; }
  Variable& Uz() { return *_Uz; }
  Variable& bz() { return *_bz; }

  Variable& Wr() { return *_Wr; }
  Variable& Ur() { return *_Ur; }
  Variable& br() { return *_br; }

  Variable& Wh() { return *_Wh; }
  Variable& Uh() { return *_Uh; }
  Variable& bh() { return *_bh; }

  virtual const Tensor& forward();

private:
  void init();

protected:
  Function  &_x;
  Function  &_h;
  Variable  *_Wz, *_Uz, *_bz;
  Variable  *_Wr, *_Ur, *_br;
  Variable  *_Wh, *_Uh, *_bh;
  Function  *_GRU;
};

// Augmented GRU function
// z(t) = Sigmoid(Wz * x(t) + Uz * h(t-1) + bz)
// r(t) = Sigmoid(Wr * x(t) + Ur * h(t-1) + br)
// p(t) = Sigmoid(Wp * x(t) + Up * h(t-1) + bp)
// q(t) = Sigmoid(Wq * x(t) + Uq * h(t-1) + bq)
// c(t) = Tanh(Wh * (p(t) . x(t)) + Uh * (r(t) . h(t-1)) + q(t) . bh)
// h(t) = z(t) . h(t-1) + (1-z(t)) . c(t)
class AGRU : public Function
{
public:
  AGRU(Graph& graph, Function& x, Function& h, int in, int out);
  AGRU(Graph& graph, Function& x, Function& h, const AGRU& other);

  Variable& Wz() { return *_Wz; }
  Variable& Uz() { return *_Uz; }
  Variable& bz() { return *_bz; }

  Variable& Wr() { return *_Wr; }
  Variable& Ur() { return *_Ur; }
  Variable& br() { return *_br; }

  Variable& Wp() { return *_Wp; }
  Variable& Up() { return *_Up; }
  Variable& bp() { return *_bp; }

  Variable& Wq() { return *_Wq; }
  Variable& Uq() { return *_Uq; }
  Variable& bq() { return *_bq; }

  Variable& Wh() { return *_Wh; }
  Variable& Uh() { return *_Uh; }
  Variable& bh() { return *_bh; }

  virtual const Tensor& forward();

private:
  void init();

protected:
  Function  &_x;
  Function  &_h;
  Variable  *_Wz, *_Uz, *_bz;
  Variable  *_Wr, *_Ur, *_br;
  Variable  *_Wp, *_Up, *_bp;
  Variable  *_Wq, *_Uq, *_bq;
  Variable  *_Wh, *_Uh, *_bh;
  Function  *_AGRU;
};

// Long Short-Term Memory (LSTM) function
// Generating Sequences WithRecurrent Neural Networks
// A. Graves, https://arxiv.org/pdf/1308.0850.pdf
// f(t) = Sigmoid(Wf * x(t) + Hf * h(t-1) + Cf * c(t-1) + bf)
// i(t) = Sigmoid(Wi * x(t) + Hi * h(t-1) + Ci * c(t-1) + bi)
// c(t) = f(t) . c(t-1) + i(t) . Tanh(Wc * x(t) + Hc * h(t-1) + bc)
// o(t) = Sigmoid(Wo * x(t) + Ho * h(t-1) + Co * c(t) + bo)
// h(t) = o(t) . Tanh(c(t))
class LSTM : public Function
{
public:
  LSTM(Graph& graph, Function& x, Function& h, Function& c, int in, int out);
  LSTM(Graph& graph, Function& x, Function& h, Function& c, const LSTM& other);

  Variable& Wf() { return *_Wf; }
  Variable& Hf() { return *_Hf; }
  Variable& Cf() { return *_Cf; }
  Variable& bf() { return *_bf; }

  Variable& Wi() { return *_Wi; }
  Variable& Hi() { return *_Hi; }
  Variable& Ci() { return *_Ci; }
  Variable& bi() { return *_bi; }

  Variable& Wo() { return *_Wo; }
  Variable& Ho() { return *_Ho; }
  Variable& Co() { return *_Co; }
  Variable& bo() { return *_bo; }

  Variable& Wc() { return *_Wc; }
  Variable& Hc() { return *_Hc; }
  Variable& bc() { return *_bc; }

  // c(t)
  Function& cell() { return *_cell; }

  virtual const Tensor& forward();

private:
  void init();

protected:
  Function &_x, &_h, &_c;
  Variable *_Wf, *_Hf, *_Cf, *_bf;
  Variable *_Wi, *_Hi, *_Ci, *_bi;
  Variable *_Wo, *_Ho, *_Co, *_bo;
  Variable *_Wc, *_Hc, *_bc;
  Function *_LSTM;
  Function *_cell;
};

// Forward Gated Unit (FGU) function
// p(t) = Sigmoid(Wp * x(t) + bp)
// q(t) = Sigmoid(Wq * x(t) + bq)
// h(t) = q(t) * Tanh(Wh * (p(t) . x(t)) + bh)
class FGU : public Function
{
public:
  FGU(Graph& graph, Function& x, int in, int out);
  FGU(Graph& graph, Function& x, const FGU& other);

  Variable& Wp() { return _Lp->W(); }
  Variable& bp() { return _Lp->b(); }

  Variable& Wq() { return _Lq->W(); }
  Variable& bq() { return _Lq->b(); }

  Variable& Wh() { return *_Wh; }
  Variable& bh() { return *_bh; }

  virtual const Tensor& forward();

private:
  void init();

protected:
  Function &_x;
  Linear   *_Lp, *_Lq;
  Variable *_Wh, *_bh;
  Function *_FGU;
};

// Sampler
class Sampler : public Function
{
public:
  Sampler(Graph& graph, Function& m, Function& s);

  virtual const Tensor& forward();

protected:
  Function &_m, &_s;
};

// Norm
class Norm : public Function
{
public:
  Norm(Graph& graph, Function& x, DTYPE g, DTYPE b);

  virtual const Tensor& forward();

protected:
  Function &_x;
  Function *_N;
  Constant *_H;
};

// Gaussian
class Gaussian : public Function
{
public:
  Gaussian(Graph& graph, Function& x, Function& m, Function& s);

  virtual const Tensor& forward();

protected:
  Function *_a; // 1 / sqrt(2*pi*s^2)
  Function *_z; // F = a * exp(z)
};

// LogGaussian
class LogGaussian : public Function
{
public:
  LogGaussian(Graph& graph, Function& x, Function& m, Function& s);

  virtual const Tensor& forward();

protected:
  Function *_a; // log(a)
  Function *_z; // log(exp(z))
};

// Hopfield
class Hopfield : public Function
{
public:
  Hopfield(Graph& graph, Function& x, DTYPE b, int size, int count);
  Hopfield(Graph& graph, Function& x, const Hopfield& other);

  // variable access
  Variable& W() { return *_W; }

  virtual const Tensor& forward();

private:
  void init(Function& x);

protected:
  DTYPE _b;
  Variable *_W;
  Function *_H;
};

// Embedding function
class Embedding : public Function
{
public:
  Embedding(Graph& graph, Constant& i, int in, int out);
  Embedding(Graph& graph, Constant& i, const Embedding& other);

  // variable access
  Variable& E() { return *_E; }

  virtual const Tensor& forward();

private:
  void init();

protected:
  Constant& _i;
  Variable* _E;
};

// No Value computed in the graph exception
class NoValueException : public std::exception
{
public:
  NoValueException() : std::exception() {}
};

// Gradient Aggregator
class Aggregator
{
public:
  virtual void aggregate(
  Tensor& g, const std::vector<Function*>& derivative) const = 0;
};

// Function Graph
class Graph : public Aggregator
{
public:
  Graph()
  {
    _aggregator = this;
  }

  virtual ~Graph()
  {
    clear();
  }

  void clear();

  // random number generator
  RNG& random() { return _rng; }

  // get function by id
  Function* function(int id) const;

  // get function by name
  Function* function(const char* name) const;

  // graph variables
  const std::vector<Variable*>& variables() const { return _vars; }

  // track function
  void keep(Function* f, const char* name = nullptr);

  // track variable
  void keep(Variable* v, const char* name = nullptr);

  // reset cache
  void recache();

  // reset gradients
  void zero_grad();

  // compute gradients
  void backward(Function& f, const Tensor& g);

  // get gradient aggregotar
  Aggregator& aggregator() const { return *_aggregator; }

  // set gradient aggregotar
  void aggregator(Aggregator& a) { _aggregator = &a; }

  // aggreagor implementation
  virtual void aggregate(
  Tensor& g, const std::vector<Function*>& derivative) const;

  ///////////////////////////////////////////
  // numerical derivative
  ///////////////////////////////////////////

  DTYPE dFdX(Function& f, Variable& x, int fr, int fc, int xr, int xc);

  Tensor dFdX(Function& f, Variable& x);

  ///////////////////////////////////////////
  // node constructors
  ///////////////////////////////////////////

  Constant* new_constant(int rows = 0, int cols = 0)
  {
    auto node = new Constant(*this, rows, cols);
    keep(node);
    return node; 
  }

  Variable* new_variable(int rows = 0, int cols = 0,
  const char* name = nullptr)
  {
    auto other = function(name);
    if (other)
    {
      return (Variable*)other;
    }
    else
    {
      auto node = new Variable(*this, rows, cols);
      keep(node, name);
      return node;
    }
  }

  Broadcast* new_broadcast(Function& x, Function& target)
  {
    auto node = new Broadcast(*this, x, target);
    keep(node);
    return node; 
  }

  Split* new_split(Function& x, int r, int c, int rows, int cols)
  {
    auto node = new Split(*this, x, r, c, rows, cols);
    keep(node);
    return node; 
  }

  Join* new_join(Function& x, Function& y, int rows, int cols)
  {
    auto node = new Join(*this, x, y, rows, cols);
    keep(node);
    return node; 
  }

  Min* new_min(Function& x, Function& y)
  {
    auto node = new Min(*this, x, y);
    keep(node);
    return node; 
  }

  Max* new_max(Function& x, Function& y)
  {
    auto node = new Max(*this, x, y);
    keep(node);
    return node; 
  }

  Linear* new_linear(Function& x, int in = 0, int out = 0,
  const char* name = nullptr)
  {
    auto other = function(name);
    if (other)
    {
      return new_linear(x, (Linear&)*other);
    }
    else
    {
      auto node = new Linear(*this, x, in, out);
      keep(node, name);
      return node;
    }
  }

  Linear* new_linear(Function& x, const Linear& other)
  {
    auto node = new Linear(*this, x, other);
    keep(node);
    return node; 
  }

  Product* new_product(Function& x, Function& y)
  {
    auto node = new Product(*this, x, y);
    keep(node);
    return node; 
  }

  Add* new_add(Function& x, Function& y)
  {
    auto node = new Add(*this, x, y);
    keep(node);
    return node; 
  }

  Sub* new_sub(Function& x, Function& y)
  {
    auto node = new Sub(*this, x, y);
    keep(node);
    return node; 
  }

  Mul* new_mul(Function& x, Function& y)
  {
    auto node = new Mul(*this, x, y);
    keep(node);
    return node; 
  }

  Power* new_power(Function& x, Function& y)
  {
    auto node = new Power(*this, x, y);
    keep(node);
    return node; 
  }

  Tanh* new_tanh(Function& x)
  {
    auto node = new Tanh(*this, x);
    keep(node);
    return node; 
  }

  Sigmoid* new_sigmoid(Function& x)
  {
    auto node = new Sigmoid(*this, x);
    keep(node);
    return node; 
  }

  ReLU* new_relu(Function& x)
  {
    auto node = new ReLU(*this, x);
    keep(node);
    return node;
  }

  Step* new_step(Function& x, DTYPE lo = 0, DTYPE hi = 1)
  {
    auto node = new Step(*this, x, lo, hi);
    keep(node);
    return node;
  }

  Dropout* new_dropout(Function& x, DTYPE rate)
  {
    auto node = new Dropout(*this, x, rate);
    keep(node);
    return node;
  }

  Softmax* new_softmax(Function& x)
  {
    auto node = new Softmax(*this, x);
    keep(node);
    return node; 
  }

  Softplus* new_softplus(Function& x)
  {
    auto node = new Softplus(*this, x);
    keep(node);
    return node; 
  }

  LogSoftmax* new_log_softmax(Function& x)
  {
    auto node = new LogSoftmax(*this, x);
    keep(node);
    return node; 
  }

  Log* new_log(Function& x)
  {
    auto node = new Log(*this, x);
    keep(node);
    return node; 
  }

  Abs* new_abs(Function& x)
  {
    auto node = new Abs(*this, x);
    keep(node);
    return node; 
  }

  Transpose* new_transpose(Function& x)
  {
    auto node = new Transpose(*this, x);
    keep(node);
    return node; 
  }

  Sum* new_sum(Function& x)
  {
    auto node = new Sum(*this, x);
    keep(node);
    return node; 
  }

  Mean* new_mean(Function& x)
  {
    auto node = new Mean(*this, x);
    keep(node);
    return node; 
  }

  GRU* new_gru(Function& x, Function& h, int in = 0, int out = 0,
  const char* name = nullptr)
  {
    auto other = function(name);
    if (other)
    {
      return new_gru(x, h, (GRU&)*other);
    }
    else
    {
      auto node = new GRU(*this, x, h, in, out);
      keep(node, name);
      return node;
    }
  }

  GRU* new_gru(Function& x, Function& h, const GRU& other)
  {
    auto node = new GRU(*this, x, h, other);
    keep(node);
    return node; 
  }

  AGRU* new_agru(Function& x, Function& h, int in = 0, int out = 0,
  const char* name = nullptr)
  {
    auto other = function(name);
    if (other)
    {
      return new_agru(x, h, (AGRU&)*other);
    }
    else
    {
      auto node = new AGRU(*this, x, h, in, out);
      keep(node, name);
      return node;
    }
  }

  AGRU* new_agru(Function& x, Function& h, const AGRU& other)
  {
    auto node = new AGRU(*this, x, h, other);
    keep(node);
    return node; 
  }

  LSTM* new_lstm(Function& x, Function& h, Function& c, int in = 0, int out = 0,
  const char* name = nullptr)
  {
    auto other = function(name);
    if (other)
    {
      return new_lstm(x, h, c, (LSTM&)*other);
    }
    else
    {
      auto node = new LSTM(*this, x, h, c, in, out);
      keep(node, name);
      return node;
    }
  }

  LSTM* new_lstm(Function& x, Function& h, Function& c, const LSTM& other)
  {
    auto node = new LSTM(*this, x, h, c, other);
    keep(node);
    return node; 
  }

  FGU* new_fgu(Function& x, int in = 0, int out = 0,
  const char* name = nullptr)
  {
    auto other = function(name);
    if (other)
    {
      return new_fgu(x, (FGU&)*other);
    }
    else
    {
      auto node = new FGU(*this, x, in, out);
      keep(node, name);
      return node;
    }
  }

  FGU* new_fgu(Function& x, const FGU& other)
  {
    auto node = new FGU(*this, x, other);
    keep(node);
    return node;
  }

  Sampler* new_sampler(Function& m, Function& s)
  {
    auto node = new Sampler(*this, m, s);
    keep(node);
    return node; 
  }

  Norm* new_norm(Function& x, DTYPE g = 1.0, DTYPE b = 0.0)
  {
    auto node = new Norm(*this, x, g, b);
    keep(node);
    return node; 
  }

  Gaussian* new_gaussian(Function& x, Function& m, Function& s)
  {
    auto node = new Gaussian(*this, x, m, s);
    keep(node);
    return node;
  }

  LogGaussian* new_log_gaussian(Function& x, Function& m, Function& s)
  {
    auto node = new LogGaussian(*this, x, m, s);
    keep(node);
    return node;
  }

  Hopfield* new_hopfield(Function& x, DTYPE b, int size, int count,
  const char* name = nullptr)
  {
    auto other = function(name);
    if (other)
    {
      return new_hopfield(x, (Hopfield&)*other);
    }
    else
    {
      auto node = new Hopfield(*this, x, b, size, count);
      keep(node, name);
      return node;
    }
  }

  Hopfield* new_hopfield(Function& x, const Hopfield& other)
  {
    auto node = new Hopfield(*this, x, other);
    keep(node);
    return node;
  }

  Embedding* new_embedding(Constant& i, int in = 0, int out = 0,
  const char* name = nullptr)
  {
    auto other = function(name);
    if (other)
    {
      return new_embedding(i, (Embedding&)*other);
    }
    else
    {
      auto node = new Embedding(*this, i, in, out);
      keep(node, name);
      return node;
    }
  }

  Embedding* new_embedding(Constant& i, const Embedding& other)
  {
    auto node = new Embedding(*this, i, other);
    keep(node);
    return node;
  }

protected:
  RNG _rng;
  Aggregator* _aggregator;
  std::vector<Function*> _nodes;
  std::vector<Variable*> _vars;
  std::unordered_map<std::string, Function*> _names;
};

} /* namespace */

#endif /*_SEEGNIFY_GRAPH_H_*/
