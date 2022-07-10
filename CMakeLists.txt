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

# Set build flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DRELEASE -O3 -g")

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed
set(CMAKE_AUTOMOC ON)
# Create code from a list of Qt designer ui files
set(CMAKE_AUTOUIC ON)

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

# Find the QtWidgets library
find_package(Qt5Widgets CONFIG REQUIRED)

# Find Image Magick++
find_package(ImageMagick COMPONENTS Magick++)
include_directories(${ImageMagick_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMAGICKCORE_HDRI_ENABLE=0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMAGICKCORE_QUANTUM_DEPTH=16")

# common
add_library (seegnify-common STATIC
main/graph.cc
main/storage.cc
main/image.cc
main/tokenizer.cc
)

# client / server
add_executable (seegnify-training
main/main.cc
main/master.cc
main/worker.cc
main/graph.pb.cc
)

# unit tests
add_executable (seegnify-unittest
main/unittest.cc
)

# full test
add_library (seegnify-fulltest SHARED
main/fulltest.cc
)

# RL
add_library (seegnify-rl STATIC
rl/env.cc
rl/gui.cc
rl/slider.cc
rl/plot.cc
)

# RL GUI
add_executable (seegnify-rl-gui
rl/main.cc
)

# SELECTOR-GAUSSIAN
add_library (example-selector-gaussian SHARED
examples/selector-gaussian.cc
)

# SELECTOR-SOFTMAX
add_library (example-selector-softmax SHARED
examples/selector-softmax.cc
)

# MNIST
add_library (example-mnist SHARED
examples/mnist.cc
)

# CIFAR
add_library (example-cifar SHARED
examples/cifar10.cc
)

# CIFAR RL
add_library (example-cifarRL SHARED
examples/cifarRL.cc
)

# MNIST RL
add_library (example-mnistRL SHARED
examples/mnistRL.cc
)

# set dependency libs
list(APPEND DL_LIBS pthread protobuf dl sndfile seegnify-common)
list(APPEND DL_LIBS PocoFoundation PocoNet)
list(APPEND DL_LIBS Qt5::Widgets Magick++-6.Q16 MagickCore-6.Q16)

# link targets
target_link_libraries(seegnify-common Qt5::Widgets)
target_link_libraries(seegnify-training ${DL_LIBS})
target_link_libraries(seegnify-unittest ${DL_LIBS})
target_link_libraries(seegnify-fulltest ${DL_LIBS})

target_link_libraries(seegnify-rl Qt5::Widgets)
target_link_libraries(seegnify-rl-gui seegnify-rl Qt5::Widgets)

target_link_libraries(example-selector-gaussian ${DL_LIBS})
target_link_libraries(example-selector-softmax ${DL_LIBS})
target_link_libraries(example-mnist ${DL_LIBS})
target_link_libraries(example-cifar ${DL_LIBS})
target_link_libraries(example-mnistRL ${DL_LIBS} seegnify-rl)
target_link_libraries(example-cifarRL ${DL_LIBS} seegnify-rl)

