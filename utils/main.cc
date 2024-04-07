/*
 * Copyright (c) 2024 Greg Padiasek
 * Distributed under the terms of the MIT License.
 * See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT
 */

#include <iostream>
#include <thread>
#include <csignal>

#include "transport.hh"
#include "graph.pb.h"

namespace seegnify {

// master routines
extern void master_init(const std::string& file);
extern void master_run(const ServerContext& ctx,
graph::Request& req, graph::Response& res);
extern void master_err(const std::exception& err, graph::Response& res);
extern void master_term();

// worker routines
extern void worker_run(const std::string& library,
const std::string& host, int port);
extern void worker_term();

// term routine
typedef void (*v_routine)();
v_routine term_routine = nullptr;

// server type
typedef ProtobufServer<graph::Request, graph::Response> GraphServer;

// server pointer for singal handling
std::shared_ptr<GraphServer> graph_server;

// signal handler
static void on_signal(int signum) {
  if (graph_server != nullptr) graph_server->stop();
  else 
  if (term_routine !=nullptr) term_routine();
}

// syntax message
void syntax(char* argv[]) {
  std::cerr << "Usage: " << argv[0] << " "
            << "master <FILE> <PORT> | "
            << "worker <HOST> <PORT> <IMPL>"
            << std::endl;
}

} /* namespace */

using namespace seegnify;

/**
 * server entry point
 */
int main(int argc, char* argv[]) {

  try {
    if (argc < 2) {
      syntax(argv);
      return 1;
    }

    // get server role
    std::string role = argv[1];

    // set signal handler
    signal(SIGINT, on_signal);

    if (role == "master") {
      if (argc != 4) {
        syntax(argv);
        return 1;
      }
      
      std::string file = argv[2];
      int port = std::stoi(argv[3]);
      std::cout << "Starting " << role << " on port " << port << std::endl;

      // start master
      master_init(file);
      graph_server = std::make_shared<GraphServer>(master_run, master_err);
      graph_server->run(port, std::thread::hardware_concurrency());
      master_term();

      std::cout << "Stopping " << role << " on port " << port << std::endl;
    }
    else
    if (role == "worker") {
      if (argc != 5) {
        syntax(argv);
        return 1;
      }

      std::string host = argv[2];
      int port = std::stoi(argv[3]);
      std::string impl = argv[4];
      std::cout << "Starting " << role << " at " 
                << host << ":" << port << std::endl;

      // start worker
      term_routine = worker_term;
      worker_run(impl, host, port);

      std::cout << "Stopping " << role << " at " 
                << host << ":" << port << std::endl;
    }
    else {
      std::cerr << "Unknown role '" << role << "'" << std::endl;
      return 3;
    }
  }
  catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 4;
  }

  return 0;
}

