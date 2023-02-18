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

#ifndef _SEEGNIFY_OPTIMIZER_H_
#define _SEEGNIFY_OPTIMIZER_H_

#include "graph.hh"

namespace seegnify {

// An overview of gradient descent optimization algorithms
// https://arxiv.org/pdf/1609.04747.pdf
// A Survey of Optimization Methods from a Machine Learning Perspective
// https://arxiv.org/pdf/1906.06821.pdf

// Simple Moving Average
// S(n) = (x(1)+...+x(n)) / n
// Numerically Stable:
// S(n) = S(n-1) + (x(n) - S(n-1)) / n
class SMA
{
public:
  SMA(DTYPE sma = 0, DTYPE N = 0) : _init(sma), _N(N) {}

  void update(const Tensor& x)
  {
    if (!_sma.size())
      _sma = Tensor::Constant(x.rows(), x.cols(), _init);

    // a small increment on float works only up to 0x1000000,
    // after that the procedure becomes EMA with alpha = 1.0 / 0x1000000
    _N++;

    _sma += (x - _sma) / _N;
  }

  const Tensor& operator()() { return _sma; }

private:
  DTYPE _N;
  DTYPE _init;
  Tensor _sma;
};

// Weighted Moving Average
// S(n) = (w(1) * x(1)+...+ w(n) * x(n)) / (w(1)+...+w(n))
// Numerically Stable:
// S(n) = S(n-1) + w(n) * (x(n) - S(n-1)) / (w(1)+...+w(n))
class WMA
{
public:
  WMA(DTYPE wma = 0, DTYPE W = 0) : _init(wma), _W(W) {}

  void update(const Tensor& x, DTYPE w)
  {
    if (!_wma.size())
      _wma = Tensor::Constant(x.rows(), x.cols(), _init);

    // a small increment on float works only up to 0x1000000,
    // after that the procedure becomes EMA with alpha = 1.0 / 0x1000000
    _W += w;

    _wma += w * (x - _wma) / (_W + EPSILON);
  }

  const Tensor& operator()() { return _wma; }

private:
  DTYPE _W;
  DTYPE _init;
  Tensor _wma;
};


// Exponetial Moving Average
// S(n) = beta * S(n-1) + (1-beta) * x(n)
// alpha = (1-beta)
// S(n) = alpha * x(n) + (1-alpha) * S(n-1)
// Numerically Stable:
// S(n) = S(n-1) + alpha * (x(n) - S(n-1))
class EMA
{
public:
  EMA(DTYPE ema = 0, DTYPE beta = 0) : _init(ema), _alpha(1-beta) {}

  void update(const Tensor& x)
  {
    if (!_ema.size())
      _ema = Tensor::Constant(x.rows(), x.cols(), _init);

    _ema += _alpha * (x - _ema);
  }

  const Tensor& operator()() { return _ema; }

private:
  DTYPE _init;
  DTYPE _alpha;
  Tensor _ema;
};

// Abstract optimizer
class Optimizer
{
public:
  virtual ~Optimizer() {} 
  virtual void update() = 0;
};

// SGD + momentum optimizer
class SGD : public Optimizer
{
public:
  SGD(const std::vector<Variable*>& vars, DTYPE lr, DTYPE beta = 0.1)
  {
    _vars = vars;
    _lr = lr;
    for (auto e: _vars)
    {
      _momentum.push_back(new EMA(0, beta));
    }
  }

  ~SGD()
  {
    for (auto ema: _momentum) delete ema;
  }

  void update()
  {
    int size = _vars.size();
    for (int i=0; i<size; i++)
    {
      auto& w = _vars[i]->value();
      auto& g = _vars[i]->gradient();
      auto& d = *_momentum[i];

      d.update(g);

      w -= _lr * d();
    }
  }

private:
  DTYPE _lr;
  std::vector<Variable*> _vars;
  std::vector<EMA*> _momentum;
};

// RMSprop optimizer
// https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
class RMSprop : public Optimizer
{
public:
  RMSprop(const std::vector<Variable*>& vars, DTYPE lr, DTYPE gamma = 0.9)
  {
    _vars = vars;
    _t = 0;
    _lr = lr;
    _gamma = gamma;
    for (auto e: _vars)
    {
      _egg.push_back(Tensor(0,0));
    }
  }

  void update()
  {
    // increment step (common for all variables)
    _t++;

    // update each variable individually
    int size = _vars.size();
    for (int i=0; i<size; i++)
    {
      auto& w = _vars[i]->value();
      auto& g = _vars[i]->gradient();
      auto& egg = _egg[i];

      if (!egg.size()) egg = g.array() * g.array();

      // update g^2 EMA
      auto gg = (g.array() * g.array()).matrix();
      egg += (1.0 - _gamma) * (gg - egg);

      // update weights
      w -= _lr * (g.array() / (egg.array() + EPSILON)).matrix();
    }
  }

private:
  DTYPE _t;
  DTYPE _lr;
  DTYPE _gamma;
  std::vector<Tensor> _egg;
  std::vector<Variable*> _vars;
};

// Adam optimizer
// https://arxiv.org/pdf/1412.6980.pdf
class Adam : public Optimizer
{
public:
  Adam(const std::vector<Variable*>& vars, DTYPE lr,
  DTYPE beta1 = 0.9, DTYPE beta2 = 0.999)
  {
    _vars = vars;
    _t = 0;
    _lr = lr;
    _beta1 = beta1;
    _beta2 = beta2;
    for (auto e: _vars)
    {
      _s1.push_back(Tensor(0,0));
      _s2.push_back(Tensor(0,0));
    }
  }

  void update()
  {
    // increment step (common for all variables)
    _t++;

    // update each variable individually
    int size = _vars.size();
    for (int i=0; i<size; i++)
    {
      auto& w = _vars[i]->value();
      auto& g = _vars[i]->gradient();
      auto& s1 = _s1[i];
      auto& s2 = _s2[i];

      if (!s1.size()) s1 = g;
      if (!s2.size()) s2 = g.array() * g.array();

      // update EMA
      auto gg = (g.array() * g.array()).matrix();
      s1 += (1.0 - _beta1) * (g - s1);
      s2 += (1.0 - _beta2) * (gg - s2);

      // comupte bias correction
      auto sc1 = s1 / (1.0 - powf(_beta1, _t));
      auto sc2 = s2 / (1.0 - powf(_beta2, _t));
      auto lr = _lr * sqrt(1.0 - powf(_beta2, _t)) / (1.0 - powf(_beta1, _t));

      // update weights
      w -= lr * (sc1.array() / ((sc2.array() + EPSILON)).sqrt()).matrix();
    }
  }

private:
  DTYPE _t;
  DTYPE _lr;
  DTYPE _beta1;
  DTYPE _beta2;
  std::vector<Tensor> _s1;
  std::vector<Tensor> _s2;
  std::vector<Variable*> _vars;
};

// AdamNC optimizer
// https://arxiv.org/abs/1811.09358
class AdamNC : public Optimizer
{
public:
  AdamNC(const std::vector<Variable*>& vars, DTYPE lr,
  DTYPE beta = 0.9, DTYPE lambda = 0.99)
  {
    _vars = vars;
    _t = 0;
    _lr = lr;
    _beta = beta;
    _lambda = lambda;
    for (auto e: _vars)
    {
      _s1.push_back(Tensor(0,0));
      _s2.push_back(Tensor(0,0));
    }
  }

  void update()
  {
    // increment step (common for all variables)
    _t++;

    auto beta1 = _beta * powf(_lambda, _t);
    auto beta2 = 1.0 - 1.0 / _t;
    auto lr = _lr / sqrtf(_t);

    // update each variable individually
    int size = _vars.size();
    for (int i=0; i<size; i++)
    {
      auto& w = _vars[i]->value();
      auto& g = _vars[i]->gradient();
      auto& s1 = _s1[i];
      auto& s2 = _s2[i];

      if (!s1.size()) s1 = g;
      if (!s2.size()) s2 = g.array() * g.array();

      // update averages
      auto gg = (g.array() * g.array()).matrix();
      s1 += (1.0 - beta1) * (g - s1);
      s2 += (1.0 - beta2) * (gg - s2);

      // update weights
      w -= lr * (s1.array() / ((s2.array() + EPSILON)).sqrt()).matrix();
    }
  }

private:
  DTYPE _t;
  DTYPE _lr;
  DTYPE _beta;
  DTYPE _lambda;
  std::vector<Tensor> _s1;
  std::vector<Tensor> _s2;
  std::vector<Variable*> _vars;
};

// Yogi optimizer
// https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization.pdf
class Yogi : public Optimizer
{
public:
  Yogi(const std::vector<Variable*>& vars, DTYPE lr,
  DTYPE beta1 = 0.9, DTYPE beta2 = 0.999)
  {
    _vars = vars;
    _t = 0;
    _lr = lr;
    _beta1 = beta1;
    _beta2 = beta2;
    for (auto e: _vars)
    {
      _s1.push_back(Tensor(0,0));
      _s2.push_back(Tensor(0,0));
    }
  }

  void update()
  {
    // increment step (common for all variables)
    _t++;

    // update each variable individually
    int size = _vars.size();
    for (int i=0; i<size; i++)
    {
      auto& w = _vars[i]->value();
      auto& g = _vars[i]->gradient();
      auto& s1 = _s1[i];
      auto& s2 = _s2[i];

      if (!s1.size()) s1 = g;
      if (!s2.size()) s2 = g.array() * g.array();

      // updage averages
      auto gg = g.array() * g.array();
      s1 += (1.0 - _beta1) * (g - s1);
      s2 += (1.0 - _beta2) * (gg * (gg - s2.array()).sign()).matrix();

      // comupte bias correction
      auto sc1 = s1 / (1.0 - powf(_beta1, _t));
      auto sc2 = s2 / (1.0 - powf(_beta2, _t));
      auto lr = _lr * sqrt(1.0 - powf(_beta2, _t)) / (1.0 - powf(_beta1, _t));

      // update weights
      w -= lr * (sc1.array() / ((sc2.array() + EPSILON)).sqrt()).matrix();
    }
  }

private:
  DTYPE _t;
  DTYPE _lr;
  DTYPE _beta1;
  DTYPE _beta2;
  std::vector<Tensor> _s1;
  std::vector<Tensor> _s2;
  std::vector<Variable*> _vars;
};

} /* namespace */

#endif /*_SEEGNIFY_OPTIMIZER_H_*/
