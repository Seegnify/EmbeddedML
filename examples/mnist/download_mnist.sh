#==============================================================================
# Copyright (c) 2024 Greg Padiasek
# Distributed under the terms of the MIT License.
# See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT
#==============================================================================

#!/bin/bash

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

