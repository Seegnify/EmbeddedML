#==============================================================================
# Copyright (c) 2024 Greg Padiasek
# Distributed under the terms of the MIT License.
# See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT
#==============================================================================

#!/bin/sh

BASE_DIR=$(dirname "$0")/..
BASE_DIR=$(readlink -f "$BASE_DIR")

protoc -I="$BASE_DIR/main" --cpp_out="$BASE_DIR/main" "$BASE_DIR/main/graph.proto"
