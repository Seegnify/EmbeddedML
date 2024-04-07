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

typedef DistTraining* (*create_callback)();
typedef void (*destroy_callback)(DistTraining*);

create_callback create = nullptr;
destroy_callback destroy = nullptr;

// command handlers

// get master graph weights
std::string get_weights()
{
  ProtobufClient<dl::graph::Request, dl::graph::Response> client;    
  client.connect(host, port);   

  std::ostringstream weights;
  std::string version;
  
  // download weights in chunks
  while (true)
  {
    dl::graph::Request req;
    dl::graph::Response res;

    // set download version
    auto get_weights = req.mutable_get_weights();
    if (version.size()) get_weights->set_version(version);
    get_weights->set_position(weights.tellp());

    client.send(req);    
    client.receive(res);
  
    if (res.has_error()) throw std::runtime_error(res.error().message());

    if (!res.has_get_weights()) 
      throw std::runtime_error("Response is missing get_weights()");

    version = res.get_weights().version();
    auto& buffer = res.get_weights().buffer();
    weights.write(buffer.data(), buffer.size());

    // get download version
    if (res.get_weights().complete()) break;
  }
  
  client.disconnect();

  // return assembled weights
  return weights.str();
}

// set master graph weights
void set_weights(const std::string& weights)
{
  ProtobufClient<dl::graph::Request, dl::graph::Response> client;    
  client.connect(host, port);

  long position = 0;
  std::string version;

  // upload weights in chunks
  while (position < weights.size())
  {
    dl::graph::Request req;
    dl::graph::Response res;

    // get chunk and upload version
    auto set_weights = req.mutable_set_weights();
    if (version.size()) set_weights->set_version(version);
    auto len = std::min<long>(weights.size() - position, MAX_PROTOBUF_SIZE);

    // set request
    set_weights->set_buffer(weights.substr(position, len));
    set_weights->set_complete(position + len >= weights.size());
    position += len;

    client.send(req);    
    client.receive(res);

    if (res.has_success()) break;
    
    if (res.has_error()) throw std::runtime_error(res.error().message());

    // get upload stream version
    if (res.has_set_weights())
      version = res.get_weights().version();
    else
      throw std::runtime_error("Response is missing set_weights()");
  }

  client.disconnect();
}

// update master graph weights
void upd_weights(const std::string& update)
{
  ProtobufClient<dl::graph::Request, dl::graph::Response> client;    
  client.connect(host, port);

  long position = 0;
  std::string version;

  // upload update in chunks
  while (position < update.size())
  {
    dl::graph::Request req;
    dl::graph::Response res;

    // get chunk and upload version
    auto upd_weights = req.mutable_upd_weights();
    if (version.size()) upd_weights->set_version(version);
    auto len = std::min<long>(update.size() - position, MAX_PROTOBUF_SIZE);

    // set request
    upd_weights->set_buffer(update.substr(position, len));
    upd_weights->set_complete(position + len >= update.size());
    position += len;

    client.send(req);
    client.receive(res);

    if (res.has_success()) break;
    
    if (res.has_error()) throw std::runtime_error(res.error().message());

    // get upload stream version
    if (res.has_upd_weights()) 
      version = res.upd_weights().version();
    else
      throw std::runtime_error("Response is missing upd_weights()");
  }

  client.disconnect();
}

// worker routines

void thread_run()
{
  auto& impl = *create();

  // init master graph
  set_weights(impl.get_weights());

  while (!done)
  {
    // get master graph
    impl.set_weights(get_weights());

    // train worker graph
    impl.episode();

    // update master graph
    upd_weights(impl.get_update());
  }

  destroy(&impl);
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
  
  ::create = (create_callback)dlsym(handle, "create");
  if (create == nullptr)
    throw std::runtime_error("Failed to locate symbol 'create'");
  ::destroy = (destroy_callback)dlsym(handle, "destroy");
  if (destroy == nullptr)
    throw std::runtime_error("Failed to locate symbol 'destroy'");

  ::host = host;
  ::port = port;

  std::vector<std::thread> pool;

  int threads = std::thread::hardware_concurrency();
  std::cout << "starting " << threads << " threads..." << std::endl;

  for (int i=0; i<threads; i++) pool.emplace_back(thread_run);

  for (auto& e: pool) e.join();
  
  dlclose(handle);
}

void worker_term()
{
  done = true;
}

} /* namespace */
