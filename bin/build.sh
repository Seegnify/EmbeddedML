#!/bin/sh

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

BASE_DIR=$(dirname "$0")/..
BASE_DIR=$(readlink -f "$BASE_DIR")

if [ -z "$BUILD_TYPE" ]; then
  BUILD_TYPE=Release
fi

# show build type
echo "Build: $BUILD_TYPE"

# create build folder
if [ ! -d "${BASE_DIR}/build" ]; then
  mkdir "${BASE_DIR}/build"
  cd "${BASE_DIR}/build"
  cmake ..
  cd -
fi

# generate makefile
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} --build ${BASE_DIR}/build

# build
cmake --build ${BASE_DIR}/build
