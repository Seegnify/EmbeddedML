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
#include <thread>
#include <mutex>
#include <atomic>

#include "transport.hh"
#include "training.hh"
#include "graph.pb.h"

namespace seegnify {

class Master : public Training
{
  public:
  Master(int id) : Training(id) {}
  void batch_train() {}
};

// master runtime data

static bool data_loaded = false;
static Master master_data(-1);
static std::mutex master_lock;
static std::string master_file;

// master command handlers

void log_status(const std::string& info)
{
  auto time = time_to_string(time_now());
  std::cout << time << " " 
            << info
            << std::endl;
}

void save_weights(const std::string& weights, const std::string& file)
{
  std::stringstream data(weights);
  auto temp_file = file + ".new";
  write_file(temp_file, data);
  if (std::rename(temp_file.c_str(), file.c_str()))
  {
    throw std::runtime_error("Failed to save weights");
  }
}

std::string load_weights(const std::string& file)
{
  std::stringstream data;
  read_file(file, data);
  return data.str();
}

void on_get_weights(const graph::GetWeights& req, graph::Response& res)
{
  if (!data_loaded) throw std::runtime_error("Server weights not loaded");

  std::lock_guard<std::mutex> lock(master_lock);
  auto response = res.mutable_get_weights(); 
  response->set_weights(master_data.get_weights());
}

void on_set_weights(const graph::SetWeights& req, graph::Response& res)
{
  // set response success
  auto response = res.mutable_success();
    
  std::lock_guard<std::mutex> lock(master_lock);

  // accept when data not set
  if (data_loaded) return;

  // set graph weights
  master_data.set_weights(req.weights());
  data_loaded = true;

  // save graph weights
  save_weights(master_data.get_weights(), master_file);
  log_status("Weights set");
}

void on_upd_weights(const graph::UpdWeights& req, graph::Response& res)
{
  // set response success
  auto response = res.mutable_success();
    
  std::lock_guard<std::mutex> lock(master_lock);

  // apply weights update
  master_data.upd_weights(req.update());

  // save graph weights
  save_weights(master_data.get_weights(), master_file);
  log_status("Weights updated");
}

// master routines

void master_init(const std::string& file)
{
  // init master file name
  master_file = file;

  try
  {
    std::lock_guard<std::mutex> lock(master_lock);

    // load stored weights
    master_data.set_weights(load_weights(master_file));
    data_loaded = true;

    // log status
    log_status("Weights loaded from " + file);
  }
  catch(std::exception& e)
  {
    std::cout << e.what() << std::endl;

    std::lock_guard<std::mutex> lock(master_lock);
    log_status("Initialized without weights");
  }
}

void master_run(const ServerContext& ctx,
graph::Request& req, graph::Response& res)
{
  if (req.has_upd_weights()) on_upd_weights(req.upd_weights(), res);
  else
  if (req.has_get_weights()) on_get_weights(req.get_weights(), res);
  else
  if (req.has_set_weights()) on_set_weights(req.set_weights(), res);
  else
  {
    throw std::runtime_error("Command Not Supported");
  }
}

void master_err(const std::exception& err, graph::Response& res)
{
  auto* errptr = res.mutable_error();
  errptr->set_status(400);
  errptr->set_message(err.what());
}

void master_term()
{
  if (!data_loaded)
  {
    std::cout << "no state saved" << std::endl;
  }
  else
  {
    std::cout << "last state saved in " << master_file << std::endl;
  }
}

} /* namespace */
