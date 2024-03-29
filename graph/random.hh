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

#ifndef _SEEGNIFY_RANDOM_H_
#define _SEEGNIFY_RANDOM_H_

#include <random>
#include <unordered_map>

namespace seegnify {

class RNG
{
public:
  RNG()
  {
    seed();
  }

  void seed()
  {
    _generator.seed(_device());
  }

  int uniform_int(int top)
  {
    std::uniform_int_distribution<int> d(0, top);
    return d(_generator);
  }

  int uniform_int(int min, int max)
  {
    std::uniform_int_distribution<int> d(min, max);
    return d(_generator);
  }

  float uniform_dec(float top)
  {
    std::uniform_real_distribution<float> d(0.0, top);
    return d(_generator);
  }

  float uniform_dec(float min, float max)
  {
    std::uniform_real_distribution<float> d(min, max);
    return d(_generator);
  }
  
  float normal_dec(float mean, float stddev)
  {
    std::normal_distribution<float> d(mean, stddev);
    return d(_generator);
  }

  template<class Iterator>
  int discrete_choice(Iterator first, Iterator last)
  {
    std::discrete_distribution<> d(first, last);
    return d(_generator);
  }

  template<class Iterator>
  void shuffle(Iterator first, Iterator last)
  {
    std::shuffle(first, last, _generator);
  }

  // selects K unique elements between first and last iterators
  template<class Iterator>
  void shuffle(Iterator first, Iterator last, int K)
  {
    int N = std::distance(first, last);
    for (int i=0; i<K && i<N-2; i++)
    {
      std::uniform_int_distribution<int> d(i, N-1);
      int j = d(_generator);
      std::swap(*(first+i), *(first+j));
    }
  }

private:
  std::random_device _device;
  std::mt19937 _generator;
};

} /* namespace */

#endif /*_SEEGNIFY_RANDOM_H_*/
