#==============================================================================
# Copyright (c) 2024 Greg Padiasek
# Distributed under the terms of the MIT License.
# See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT
#==============================================================================

#!/bin/sh

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
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ${BASE_DIR}/build

# build
cmake --build ${BASE_DIR}/build
