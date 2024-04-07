#==============================================================================
# Copyright (c) 2024 Greg Padiasek
# Distributed under the terms of the MIT License.
# See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT
#==============================================================================

#!/bin/bash

# create data folder
mkdir -p data/cifar10
cd data/cifar10

# download data
if [ ! -f ./cifar-10-binary.tar.gz ]; then
  wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
fi
if [ ! -f ./cifar-10-batches-bin/data_batch_1.bin ]; then
  tar xvzf cifar-10-binary.tar.gz
fi

