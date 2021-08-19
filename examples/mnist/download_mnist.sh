#!/bin/bash

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

get_mnist()
{
  if [ ! -f "mnist.zip" ]; then
    wget https://data.deepai.org/mnist.zip
  fi
  if [ ! -f "$1.gz" ]; then
    unzip -o mnist.zip
  fi
  if [ ! -f "$1" ]; then
      gunzip -vc "$1.gz" > "$1"
  fi
}

# create data folder
mkdir -p data/mnist
cd data/mnist

# download data
get_mnist train-images-idx3-ubyte
get_mnist train-labels-idx1-ubyte
get_mnist t10k-images-idx3-ubyte
get_mnist t10k-labels-idx1-ubyte

