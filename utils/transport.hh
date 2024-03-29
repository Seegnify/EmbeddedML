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

#ifndef _SEEGNIFY_TRANSPORT_H_
#define _SEEGNIFY_TRANSPORT_H_

#include <Poco/Net/SocketAddress.h>
#include <Poco/Net/ServerSocket.h>
#include <Poco/Net/TCPServerConnectionFactory.h>
#include <Poco/Net/TCPServerConnection.h>
#include <Poco/Net/TCPServerParams.h>
#include <Poco/Net/TCPServer.h>
#include <Poco/Net/SocketStream.h>
#include <Poco/Net/StreamSocket.h>
#include <Poco/Condition.h>
#include <Poco/Mutex.h>
#include <sstream>

#include "storage.hh"

namespace seegnify {

class ServerContext {
public:
  ServerContext(const Poco::Net::StreamSocket& socket) {
    const auto& address = socket.peerAddress();
    _address = address.host().toString();
    _port = address.port();
  }

  inline const std::string& peer_addr() const { return _address; }
  inline const uint16_t peer_port() const { return _port; }

private:
  std::string _address;
  uint16_t _port;
};

template <typename Request, typename Response>
class ProtobufServer {
public:

  ProtobufServer(void(*handlerR)(const ServerContext&, Request&, Response&),
    void(*handlerE)(const std::exception&, Response&)) {
    _handlerR = handlerR;
    _handlerE = handlerE;
  }

  void run(uint16_t port, uint16_t threads) {
    // create server socket
    Poco::Net::SocketAddress address("0.0.0.0", port);
    Poco::Net::ServerSocket socket(address);
    socket.setNoDelay(true);

    // create multi-threaded server
    Poco::ThreadPool pool(1, threads);
    Poco::Net::TCPServerParams::Ptr params = new Poco::Net::TCPServerParams();
    params->setMaxQueued(128);
    params->setMaxThreads(threads);
    params->setThreadIdleTime(60);
    Poco::Net::TCPServer server(
      new ConnectionFactory(_handlerR, _handlerE), pool, socket);

    // start listening thread
    server.start();

    // wait indefinitely
    Poco::Mutex mtx;
    Poco::Mutex::ScopedLock lock(mtx);
    _run_condition.wait(mtx);

    // stop listening thread
    server.stop();
  }

  void stop() {
    _run_condition.signal();
  }

protected:

  // server connection factory
  class ConnectionFactory : public Poco::Net::TCPServerConnectionFactory {
    public:
      ConnectionFactory(void(*handlerR)(
        const ServerContext&, Request&, Response&),
        void(*handlerE)(const std::exception&, Response&)) {
        _handlerR = handlerR;
        _handlerE = handlerE;
      }

      Poco::Net::TCPServerConnection* createConnection(
        const Poco::Net::StreamSocket& socket) {
        return new ConnectionHandler(_handlerR, _handlerE, socket);
      }

    private:
      void(*_handlerR)(const ServerContext&, Request&, Response&);
      void(*_handlerE)(const std::exception&, Response&);
  };

  // server connection handlerR
  class ConnectionHandler : public Poco::Net::TCPServerConnection {
    public:
      ConnectionHandler(void(*handlerR)(
        const ServerContext&, Request&, Response&),
        void(*handlerE)(const std::exception&, Response&),
        const Poco::Net::StreamSocket& socket) :
        Poco::Net::TCPServerConnection(socket) {
          _handlerR = handlerR;
          _handlerE = handlerE;
        }

      void run() {
        auto& s = socket();
        Poco::Net::SocketInputStream input(s);
        Poco::Net::SocketOutputStream output(s);
        ServerContext context(s);
        try {
          Request request;
          Response response;

          while(read_pb(request, input)) {
            _handlerR(context, request, response);
            write_pb(response, output);
            output.flush();
            request.Clear();
            response.Clear();
          }
        }
        catch (std::exception& e) {
          on_error(e, output);
        }
      }

    protected:
      void on_error(const std::exception& e, std::ostream& output) {
        try {
          Response response;
          _handlerE(e, response);
          write_pb(response, output);
        }
        catch (std::exception& e) {
          std::cout << "Exception in session thread: " << e.what() << std::endl;
        }
      }

    private:
      void(*_handlerR)(const ServerContext&, Request&, Response&);
      void(*_handlerE)(const std::exception&, Response&);
  };

private:

  void(*_handlerR)(const ServerContext&, Request&, Response&);
  void(*_handlerE)(const std::exception&, Response&);
  Poco::Condition _run_condition;
};

template <typename Request, typename Response>
class ProtobufClient {
public:

  ~ProtobufClient() {
    disconnect();
  }

  void connect(const std::string& host, uint32_t port) {
    connect(host, std::to_string(port));
  }

  void disconnect() {
      _stream.close();
  }

  void connect(const std::string& host, const std::string& service) {
    disconnect();

    Poco::Net::SocketAddress address(host, service);
    _stream.connect(address);
    if (_stream.impl() == NULL || !_stream.impl()->initialized()) {
      std::ostringstream error;
      error << "Failed to connect to '" << host << ":" << service << "'";
      throw std::runtime_error(error.str());
    }
  }

  void send(const Request& request) {
    Poco::Net::SocketStream stream(_stream);
    write_pb(request, stream);
  }

  void receive(Response& response) {
    Poco::Net::SocketStream stream(_stream);
    read_pb(response, stream);
  }

private:

  Poco::Net::StreamSocket _stream;
};

} /* namespace */

#endif /* _SEEGNIFY_TRANSPORT_H_ */
