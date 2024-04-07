/*
 * Copyright (c) 2024 Greg Padiasek
 * Distributed under the terms of the MIT License.
 * See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT
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
  void episode() {/* no-op */}
};

// master runtime data

static bool data_loaded = false;
static Master master_data;
static std::mutex master_lock;
static std::string master_file;

// master command handlers

void log_status(const std::string& info)
{
  auto time = dl::time_to_string(dl::time_now());
  std::cout << time << " " 
            << info
            << std::endl;
}

void save_weights(const std::string& weights, const std::string& file)
{
  std::stringstream data(weights);
  dl::write_file(file, data);
}

std::string load_weights(const std::string& file)
{
  std::stringstream data;
  dl::read_file(file, data);
  return data.str();
}

void save_chunk(const std::string& file, const std::string& chunk)
{
  std::stringstream data(chunk);
  dl::write_chunk(file, data);
}

std::string load_chunk(const std::string& file, long position, bool& end)
{
  std::stringstream data;
  dl::read_chunk(file, data, position, MAX_PROTOBUF_SIZE);
  end = position + MAX_PROTOBUF_SIZE >= dl::file_size(file);
  return data.str();
}

std::string new_version()
{
  std::string version(10, 0);
  for (auto& e: version) e = 'A' + random() % ('Z' - 'A');
  return version;
}

std::string stream_path(const std::string& version)
{
  std::ostringstream path;
  path << "stream/" << version;
  return path.str();
}

void on_get_weights(const dl::graph::GetWeights& req, dl::graph::Response& res)
{
  if (!data_loaded) throw std::runtime_error("Server weights not loaded");

  std::lock_guard<std::mutex> lock(master_lock);
  auto response = res.mutable_get_weights();

  // create new stream
  auto version = (req.has_version()) ? req.version() : new_version();
  auto storage = stream_path(version);
  if (!req.has_version()) save_weights(master_data.get_weights(), storage);

  // read weights chunk
  auto complete = false;
  auto buffer = load_chunk(storage, req.position(), complete);

  // delete the stream
  if (complete) remove(storage.data());

  // set chunk response
  response->set_version(version);
  response->set_buffer(buffer);
  response->set_complete(complete);
}

void on_set_weights(const dl::graph::SetWeights& req, dl::graph::Response& res)
{    
  // set response success
  auto response = res.mutable_success();

  std::lock_guard<std::mutex> lock(master_lock);

  // accept when data not set
  if (data_loaded) return;

  // save chunk to stream
  auto version = (req.has_version()) ? req.version() : new_version();
  auto storage = stream_path(version);
  std::istringstream buffer(req.buffer());
  dl::write_chunk(storage, buffer);
  
  if (req.complete())
  {
    // set graph weights
    master_data.set_weights(load_weights(storage));
    data_loaded = true;

    // delete the stream
    remove(storage.data());

    // save graph weights
    save_weights(master_data.get_weights(), master_file);
    log_status("Weights set");
  }
  else
  {
    // set version response
    auto response = res.mutable_set_weights();
    response->set_version(version);
  }
}

void on_upd_weights(const dl::graph::UpdWeights& req, dl::graph::Response& res)
{    
  std::lock_guard<std::mutex> lock(master_lock);

  // save chunk to stream
  auto version = (req.has_version()) ? req.version() : new_version();
  auto storage = stream_path(version);
  std::istringstream buffer(req.buffer());
  dl::write_chunk(storage, buffer);

  if (req.complete())
  {
    // apply weights update
    master_data.upd_weights(load_weights(storage));

    // delete the stream
    remove(storage.data());

    // save graph weights
    save_weights(master_data.get_weights(), master_file);
    log_status("Weights updated");
 
    // set response success
    auto response = res.mutable_success();
  }
  else
  {
    // set version response
    auto response = res.mutable_upd_weights();
    response->set_version(version);
  }
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
dl::graph::Request& req, dl::graph::Response& res)
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

void master_err(const std::exception& err, dl::graph::Response& res)
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
