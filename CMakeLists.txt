#
# Copyright 2020-2021 Greg Padiasek and Seegnify <http://www.seegnify.org>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# CMakeLists files in this project can
# refer to the root source directory and root bnary directory of the project as
# respectively ${SEEGNIFY_SOURCE_DIR} and ${SEEGNIFY_BINARY_DIR}.
cmake_minimum_required (VERSION 2.8.11)
project (SEEGNIFY)

# Build type
# set(CMAKE_BUILD_TYPE Release)

# CMake policies
cmake_policy(SET CMP0100 OLD) 

# Set build flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DRELEASE -O3 -g")

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Show compiler stdout
set(CMAKE_VERBOSE_MAKEFILE ON)

# Enable C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fvisibility=hidden")

# Skip rpath settings
set(CMAKE_SKIP_RPATH TRUE)

# Define additional include dirs
include_directories(/usr/include/eigen3)
include_directories(examples/mnist/include)
include_directories(examples/cifar10/include)

# Find Image Magick++
find_package(ImageMagick COMPONENTS Magick++)
include_directories(${ImageMagick_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMAGICKCORE_HDRI_ENABLE=0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMAGICKCORE_QUANTUM_DEPTH=16")

# Find ZLib
find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

# common
add_library (seegnify-common STATIC
graph/graph.cc
utils/storage.cc
utils/image.cc
utils/imageFP.cc
utils/graph.pb.cc
)

# library build options
target_compile_options(
    seegnify-common PRIVATE
    -fPIC
)

# client / server
add_executable (seegnify-training
utils/main.cc
utils/master.cc
utils/worker.cc
)

# unit tests
add_executable (seegnify-unittest
utils/unittest.cc
utils/rlenv.cc
)

# full test
add_library (seegnify-fulltest SHARED
utils/fulltest.cc
)

# dopamine
add_executable (test-dopamine
examples/dopamine-test.cc
)

# test regression
add_executable (test-regression
examples/regression-test.cc
examples/regression.cc
)

# test transformer
add_executable (test-transformer
examples/transformer-test.cc
examples/transformer.cc
external/cnpy/cnpy.cpp
)

# train TRANSFORMER
add_library (example-transformer SHARED
examples/transformer.cc
external/cnpy/cnpy.cpp
)

# train MNIST
add_library (example-mnist SHARED
examples/mnist.cc
)

# train CIFAR
add_library (example-cifar SHARED
examples/cifar10.cc
)

# set dependency libs
list(APPEND DL_LIBS seegnify-common)
list(APPEND DL_LIBS pthread protobuf dl)
list(APPEND DL_LIBS PocoFoundation PocoNet)
list(APPEND DL_LIBS Magick++-6.Q16 MagickCore-6.Q16)

# link targets
target_link_libraries(seegnify-common)
target_link_libraries(seegnify-training ${DL_LIBS})
target_link_libraries(seegnify-unittest ${DL_LIBS})
target_link_libraries(seegnify-fulltest ${DL_LIBS})

target_link_libraries(test-transformer ${DL_LIBS} ${ZLIB_LIBRARIES})
target_link_libraries(test-dopamine ${DL_LIBS})
target_link_libraries(test-regression ${DL_LIBS})

target_link_libraries(example-transformer ${DL_LIBS})
target_link_libraries(example-mnist ${DL_LIBS})
target_link_libraries(example-cifar ${DL_LIBS})

