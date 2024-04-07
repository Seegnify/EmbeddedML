#==============================================================================
# Copyright (c) 2024 Greg Padiasek
# Distributed under the terms of the MIT License.
# See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT
#==============================================================================

#!/bin/sh

pgrep -f seegnify-training | while read ST_PID
do
  kill -TERM $ST_PID
done
