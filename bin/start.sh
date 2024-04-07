#==============================================================================
# Copyright (c) 2024 Greg Padiasek
# Distributed under the terms of the MIT License.
# See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT
#==============================================================================

#!/bin/sh

start() {
  PORT=$1
  IMPL=$2
  CMD=./build/seegnify-training
  LIB=./build/lib$IMPL.so
  echo Starting master $IMPL at port $PORT
  nice -n 10 $CMD master $IMPL.graph $PORT >> $IMPL.master.log 2>&1 &
  sleep 1
  echo Starting worker $IMPL at port $PORT
  nice -n 10 $CMD worker 127.0.0.1 $PORT $LIB >> $IMPL.client.log 2>&1 &
  #valgrind $CMD worker 127.0.0.1 $PORT $LIB >> $IMPL.client.log 2>&1 &
}

ulimit -c unlimited

start 9090 example-transformer
#start 9191 example-regression
