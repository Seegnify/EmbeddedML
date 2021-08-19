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

