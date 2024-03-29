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

#ifndef _SEEGNIFY_UNITTEST_H_
#define _SEEGNIFY_UNITTEST_H_

#include <iostream>
#include <sstream>

namespace seegnify {

class UnitTest {
  public:
    UnitTest(const char* text) {
      _done = false;
      _line = -1;
      _file = nullptr;
      std::cout << "test [" << text << "]: " << std::flush;
    }

    ~UnitTest() {
      end_log();
    }

    void finish() {
      _done = true;
    }

    void set_position(int line, const char* file) {
      _line = line;
      _file = file;
    }

    void log(const char* msg) {
      if (_file != nullptr) {
        _log << _line << ":" << _file << " ";
      }
      _log << msg << std::endl;
    }

    void end_log()
    {
      std::string log_str = _log.str();
      if (!_done) {
        std::cout << "UNFINISHED" << std::endl;
      }
      else
      if (log_str.size() == 0) {
        std::cout << "OK" << std::endl;
      }
      else {
        std::cout << "FAILED" << std::endl;
      }
      std::cout << log_str;
    }

  private:
    bool _done;
    std::ostringstream _log;
    int _line;
    const char* _file;
};

#define TEST_BEGIN(text) \
  {\
    UnitTest _ut(text);\
    try\
    {

#define TEST_END() \
      _ut.finish(); \
    }\
    catch(std::exception& e)\
    {\
      _ut.log(e.what());\
    }\
  }

#define ASSERT(expr) \
  _ut.set_position(__LINE__, __FILE__);\
  if (!(expr)) _ut.log("assert failure");\
  _ut.set_position(-1, nullptr);

} /* namespace */

#endif /*_SEEGNIFY_UNITTEST_H_*/

