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

#include <chrono>
#include <thread>

#include "main/optimizer.hh"
#include "utils/training.hh"
#include "utils/storage.hh"
#include "utils/rlenv.hh"

#include "cifar/cifar10_reader.hpp"
#include "cifarRL.hh"

// training params
#define BATCH   1000
#define LRATE   1e-3
#define GAMMA   1.0
#define EXPLORE 0.03

// progress log
#define VALSTEPS 100000

////////////////////////////////
// RL training environment
////////////////////////////////
class CIFARRLEnv : public RLEnv
{
public:
  enum ACTION
  {
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_FORWARD,
    ACTION_BACKWARD,
    ACTION_ZOOM_IN,
    ACTION_ZOOM_OUT,
  };

  CIFARRLEnv()
  {
    finished = false;
    last_action = NUM_OF_ACTIONS;
    set_view_size(VIEW_ROWS, VIEW_COLS);
  }

  void set_label(int l)
  {
    label = l;
  }

  bool is_correct()
  {
    return last_action - NUM_OF_ACTIONS == label;
  }

  uint16_t get_actions_count()
  {
    return NUM_OF_ACTIONS;
  }

  bool is_episode_finished()
  {
    return finished || action_step > NUM_OF_ACTIONS || total_reward < MIN_REWARD;
  }

  float make_action(uint16_t action)
  {
    last_action = action;

    switch(action)
    {
      case ACTION_UP:         action_up(); break;
      case ACTION_DOWN:       action_down(); break;
      case ACTION_LEFT:       action_left(); break;
      case ACTION_RIGHT:      action_right(); break;
      case ACTION_FORWARD:    action_forward(); break;
      case ACTION_BACKWARD:   action_backward(); break;
      case ACTION_ZOOM_IN:    action_zoom_in(); break;
      case ACTION_ZOOM_OUT:   action_zoom_out(); break;
    };

    if (last_action >= NUM_OF_ACTIONS) finished = true;

    auto r = get_reward();
    total_reward += r;
    return r;
  }
  
protected:
  virtual void reset()
  {
    slice = rand() % slices;

    auto center_x = full_cols / 2;
    auto center_y = full_rows / 2;
    auto max_dist = full_cols / 4;

    x = center_x + rand() % (2 * max_dist) - max_dist;
    y = center_y + rand() % (2 * max_dist) - max_dist;

    total_reward = 0;
    label = NUM_OF_ACTIONS;
    last_action = NUM_OF_ACTIONS;
  }

  // [0,1] reward for loss: -log(p) a r -log(p-1) a (1-r)
  virtual float get_reward()
  {
    return is_correct();
  }

private:
  bool finished;
  float total_reward;
  uint16_t last_action;
  uint16_t label;
};

////////////////////////////////
// CIFAR RL training
////////////////////////////////
class CIFARRLClient : public Training
{
public:
  CIFARRLClient(int worker) : Training(worker)
  {
    std::cout << "CIFAR_RL training " << worker + 1 << std::endl;

    // load data
    _data = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>
    ("./data/cifar10/cifar-10-batches-bin");

    // get graph
    Graph& g = graph();

    // ceate RL env
    _env = new CIFARRLEnv();

    // ceate RL model
    _mod = new CIFARRLModel(g);

    // optimizer
    _opt = new Adam(g.variables(), LRATE);

    // state
    _episode = 0;
    _frames = 0;
    
    std::cout << " training_images=" << _data.training_images.size()
              << " training_labels=" << _data.training_labels.size()
              << " test_images=" << _data.test_images.size()
              << " test_labels=" << _data.test_labels.size()
              << " env rows=" << VIEW_ROWS
              << " env rows=" << VIEW_COLS
              << " input size=" << INPUT
              << " hidden size=" << HIDDEN
              << " output size=" << OUTPUT
              << std::endl;

    // loss over time
    for (int t=0; t<STEPS; t++)
    {      
      // time step action and reward
      auto& a = *g.new_constant(OUTPUT, 1);
      auto& r = *g.new_constant(1, 1);
      _action.push_back(&a);
      _reward.push_back(&r);

      // model policy
      auto& p = _mod->policy(t);

      // selected output
      //auto& s = *g.new_log(*g.new_sum(p * a)) * r;
      auto& s = *g.new_sum(*g.new_log((1 - EPSILON) * p + EPSILON) * a) * r;

      // not-selected output
      //auto& u = *g.new_log(*g.new_sum((1 - p) * a)) * (1 - r);
      auto& u = *g.new_sum(*g.new_log((1 - EPSILON) * (1 - p) + EPSILON) * a) * (1 - r);

      // time step loss
      auto& l = - s - u;

      // cumulative loss
      if (t == 0)
      {
        // use straight time loss value in the first loss term
        _loss.push_back(&l);
      }
      else
      {
        // apply straight cumulative loss
        auto& L = *_loss.back();
        _loss.push_back( &(L + l) );
      }
    }

    // data index
    _batch.resize(_data.training_images.size());
    for (int i=_data.training_images.size()-1; i>=0; i--) _batch[i] = i;
  }

  ~CIFARRLClient()
  {
    delete _opt;
    delete _env;
    delete _mod;
  }

  std::vector<uint8_t> cifar_to_rgb(const std::vector<uint8_t>& image)
  {
    // convert ciphar to RGB format
    auto pix = image.data();
    std::vector<uint8_t> rgb(image.size(), 0);

    for (int r=0; r<DATA_ROWS; r++)
    for (int c=0; c<DATA_COLS; c++) 
    for (int channel=0; channel<3; channel++) // RGB
    {
      auto rgb_index = r * DATA_COLS * 3 + c * 3 + channel;
      auto pix_index = channel * DATA_COLS * DATA_ROWS + r * DATA_COLS + c;
      rgb[rgb_index] = pix[pix_index];
    }

    return rgb;
  }

  // set input
  void set_input(Tensor& in, const Image& image)
  {
    auto imageI = image.data();
    std::vector<DTYPE> imageF(INPUT, 0);
    for (int i=0; i<INPUT; i++) imageF[i] = imageI[i];

    in.block(0,0, INPUT,1) = TensorMap(imageF.data(), INPUT, 1);

    in.block(0,0, INPUT,1) /= 255;
  }

  // set target
  void set_target(Tensor& action, int a, Tensor& reward, float r)
  {
    action.setZero();
    action(a,0) = 1;
    reward(0,0) = r;
  }

  // get output
  int get_action(const Tensor& out, float epsilon = 0.0)
  {
    // get stochastic output form the network
    if (0.0 < epsilon)
    {
      return graph().random().discrete_choice(out.data(), out.data() + out.size());
    }

    // get deterministic output form the network
    int max_row = -1;
    int max_col = -1;
    out.maxCoeff(&max_row, &max_col);
    return max_row;
  }

  float episode(const std::vector<uint8_t>& image, int label, bool train)
  {
    // clear graph cache
    graph().recache();

    // data sample
    auto sample = cifar_to_rgb(image);
    auto explore = (train) ? EXPLORE : 0;

    // new episode
    _env->new_episode();
    _env->set_full_rgb(sample.data(), 1, DATA_ROWS, DATA_COLS);
    _env->set_label(label);

    // loop in time
    _frames = 0;
    std::vector<int> actions;
    std::vector<DTYPE> rewards;
    for (int t=0; t<STEPS; t++)
    {
      _frames++;
      auto& input = _mod->input(t);
      auto& policy = _mod->policy(t);

      // get action from policy
      auto view = _env->get_view_rgb();
      set_input(input.value(), view);
      actions.push_back(get_action(policy(), explore));

      // apply action
      rewards.push_back(_env->make_action(actions.back()));
      if (_env->is_episode_finished()) break;
    }

    if (train)
    {
      auto total = std::accumulate(rewards.begin(), rewards.end(), 0);
      //if (total == 0) abort();

      // discount rewards
      rewards = discount_reward(rewards, GAMMA);

      // apply action and reward in each time step
      for (int t=0; t<_frames; t++)
      {
        // restore references
        auto& action = *_action[t];
        auto& reward = *_reward[t];

        set_target(action.value(),    actions[t],
                   reward.value(),    rewards[t]);
      }

      // get current loss
      auto& loss = *_loss[_frames-1];
      loss();

      // compute gradients
      graph().backward(loss, Tensor::Constant(1,1,1));

      // update state
      _episode++;
    }

    return _env->is_correct();
  }

  virtual void batch_train()
  {
    // train on batch of episodes
    graph().random().shuffle(_batch.begin(), _batch.end());
    float success = 0;
    for (int i=0; i<BATCH; i++)
    {
      int index = _batch[i];
      auto& image = _data.training_images[index];
      auto label = _data.training_labels[index];
      success += episode(image, label, true);
    }

    // log status
    if (worker() == 0)
    {
      int index = _batch[BATCH-1];
      int label = _data.training_labels[index];
      auto& image = _data.training_images[index];
      auto& action = _action[_frames-1]->forward();
      auto loss = _loss[_frames-1]->forward()(0);
      int guess = get_action(action) - _env->get_actions_count();

      std::cout << "batch " << _episode / BATCH
                << ", accuracy " << success / BATCH
                << ", frames " << _frames
                << ", label " << label
                << ", guess " << guess
                << ", loss " << loss
                << std::endl << std::flush;
    }

    // update weights and reset grads
    _opt->update();
    graph().zero_grad();

    // validate on test data
    if (worker() == 0 && _episode % VALSTEPS == 0)
    {
      int success = 0;

      // run batch of episodes
      int size = _data.test_images.size();
      for (int i=0; i<size; i++)
      {
        auto& image = _data.test_images[i];
        auto label = _data.test_labels[i];
        success += episode(image, label, false);

        if ((i+1) % 100 == 0)
        {
          std::cout << "validation accuracy "
                    << (i+1) << " / " << size << " : "
                    << (float)success / (float)(i+1)
                    << std::endl << std::flush;
        }
      }

      std::cout << "final validation accuracy "
                << (float)success / (float)size
                << std::endl << std::flush;
    }
  }

private:
  std::vector<Constant*> _action;
  std::vector<Constant*> _reward;
  std::vector<Function*> _loss;
  CIFARRLEnv *_env;
  CIFARRLModel* _mod;
  Optimizer* _opt;
  uint32_t _episode;
  uint32_t _frames; 
  std::vector<int> _batch;

  // cifar data
  cifar::CIFAR10_dataset<std::vector, std::vector<uint8_t>, uint8_t> _data;
};
///////////////////////////////////
extern "C" { // export C signatures
///////////////////////////////////

DLL_EXPORT Training* create(int idx)
{
  return new CIFARRLClient(idx);
}

DLL_EXPORT void destroy(Training* ptr)
{
  delete (CIFARRLClient*)ptr;
}

///////////////////////////////////
} // export C signatures
///////////////////////////////////

