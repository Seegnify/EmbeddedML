/*
 * Copyright (c) 2024 Greg Padiasek
 * Distributed under the terms of the MIT License.
 * See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT
 */

#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>

#include <dlfcn.h>

#include "transport.hh"
#include "training.hh"
#include "graph.pb.h"

namespace seegnify {

// worker runtime data

bool done = false;
std::string host;
int port = -1;

typedef Training* (*create_callback)(int);
typedef void (*destroy_callback)(Training*);

create_callback create = nullptr;
destroy_callback destroy = nullptr;

// command handlers

// get master graph weights
std::string get_weights()
{
  graph::Request req;
  graph::Response res;

  auto get_weights = req.mutable_get_weights();

  ProtobufClient<graph::Request, graph::Response> client;    
  client.connect(host, port);    
  client.send(req);    
  client.receive(res);
  client.disconnect();
  
  if (res.has_error()) throw std::runtime_error(res.error().message());
  
  auto master_weights = res.get_weights().weights();
  
  return master_weights;
}

// set master graph weights
void set_weights(const std::string& weights)
{
  graph::Request req;
  graph::Response res;

  auto set_weights = req.mutable_set_weights();
  set_weights->set_weights(weights);

  ProtobufClient<graph::Request, graph::Response> client;    
  client.connect(host, port);
  client.send(req);    
  client.receive(res);
  client.disconnect();
  
  if (res.has_error()) throw std::runtime_error(res.error().message());
}

// update master graph weights
void upd_weights(const std::string& update)
{
  graph::Request req;
  graph::Response res;

  auto upd_weights = req.mutable_upd_weights();
  upd_weights->set_update(update);

  ProtobufClient<graph::Request, graph::Response> client;    
  client.connect(host, port);
  client.send(req);    
  client.receive(res);
  client.disconnect();
  
  if (res.has_error()) throw std::runtime_error(res.error().message());
}

// worker routines

void thread_run(int worker)
{
  try
  {
    auto& impl = *create(worker);

    // init master graph
    set_weights(impl.get_weights());

    while (!done)
    {
      // get master graph
      impl.set_weights(get_weights());

      // train worker graph
      impl.batch_train();

      // update master graph
      upd_weights(impl.get_update());
    }

    destroy(&impl);
  }
  catch (std::exception& e)
  {
    std::cerr << "Worker " << worker << " exception: "
              << e.what() << std::endl;
  }
}

void worker_run(const std::string& impl, const std::string& host, int port)
{  
  void* handle = dlopen(impl.c_str(), RTLD_LAZY);
  if (handle == nullptr)
  {
    std::ostringstream log;
    log << "Failed to load libary '" << impl << "'";
    throw std::runtime_error(log.str());
  }
  
  seegnify::create = (create_callback)dlsym(handle, "create");
  if (create == nullptr)
    throw std::runtime_error("Failed to locate symbol 'create'");
  seegnify::destroy = (destroy_callback)dlsym(handle, "destroy");
  if (destroy == nullptr)
    throw std::runtime_error("Failed to locate symbol 'destroy'");

  seegnify::host = host;
  seegnify::port = port;

  std::vector<std::thread> pool;

  int threads = std::thread::hardware_concurrency();
  std::cout << "starting " << threads << " threads..." << std::endl;

  for (int i=0; i<threads; i++) pool.emplace_back(thread_run, i);

  for (auto& e: pool) e.join();
  
  dlclose(handle);
}

void worker_term()
{
  done = true;
}

} /* namespace */
