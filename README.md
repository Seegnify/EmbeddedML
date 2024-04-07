# Seegnify

[Seegnify](https://seegnify.org/) is a machine learning library that runs anywhere.

## Main Features

  * Written in C++ for performance, portability and usability
  * Automatic differentiation for training
  * SGD, Adam, AdamNC, Yogi and RMSprop optimizers
  * Extendable generic tensor with default Eigen backend
  * High-performance computing with BLAS and OpenMP
  * No third-party dependencies for training or inference
  * Out-of-the-box distributed training option

## More About Seegnify

<!-- toc -->

- [Project Structure](#project-structure)
  - [Dependencies](#dependencies)
- [Installation](#installation)
  - [From Source](#from-source)
  - [Unit Test](#unit-test)
  - [Full Test](#full-test)
- [Examples](#examples)
- [Training](#training)
- [Android Support](#android-support)

<!-- tocstop -->

### Project Structure

| Component | Description                                           |
| --------- | ----------------------------------------------------- |
| bin       | Build and run-time control scripts for convenience    |
| main      | Deep-Learning graph with automatic differentiation    |
| utils     | Utilities supporting distributed learning and testing |
| external  | External source code for data IO and run-time         |
| examples  | Neural Network models and training examples           |

### Dependencies

| Component | Eigen* | Protobuf | POCO | ImageMagick | SndFile | ZLib |
| --------- | -------| -------- | ---- | ----------- | ------- | ---- |
| bin       | no     | no       | no   | no          | no      | no   |
| main      | yes    | no       | no   | no          | no      | no   |
| utils     | yes    | yes      | yes  | yes         | yes     | no   |
| external  | yes    | no       | no   | no          | no      | yes  |
| examples  | yes    | yes      | yes  | yes         | no      | no   |

(*) Eigen is a header-only template library

## Installation

### From Source

Compile the binaries using provided CMake wrapper script:

```bash
./bin/build.sh
```

### Unit test

Executed all unit tests:

```bash
./build/seegnify-unittest
```

### Full test

Start a full end-to-end distributed training test:

```bash
./bin/start.sh
```

Monitor training progress:

```bash
tail -f worker.log
```

Stop the worker and the master processes:

```bash
./bin/stop.sh
```

## Examples

The following training code examples can be found in the `examples` folder:

| Example     | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| MNIST       | Model definition, training and validation on MNIST dataset   |
| MINST RL    | Reinforcement learning on MNIST dataset                      |
| CIFAR10     | Model definition, training and validation on CIFAR10 dataset |
| CIFAR10 RL  | Reinforcement learning on CIFAR10 dataset                    |
| Regression  | Linear regression in single and multi-process modes          |
| Transformer | Two compatible Transformer implemenations in C++ and Python  |

## Training

A complete reference implementation for single or multi-process training can
be found in the `examples` folder. To train in a single process example, run the
training loop within a single executable such as `test-regression`. To train in
a multi-process mode, compile the model as a shared object such as 
`libexample-regression.so` and start the master and worker processes as below.
To train in distributed mode, run the workers on multiple machines and point 
them to the server machine hosting the master process.

In the example below, the master process listens to workers on port 2020. The
workers connect to `localhost` on port `2020`. By default, a single worker
utilizes all available cores and assigns one training instance to each core.

Start the master process on port 2020 and store the model in my.graph:

```bash
./build/seegnify-training master my.graph 2020
```

Start the worker process with your compiled model and connect to master:

```bash
./build/seegnify-training worker 127.0.0.1 2020 ./build/libexample-regression.so
```

Stop the training by sending SIGINT (Ctrl-C) signal to the master process.

## Android Support

To build the library for Android simply copy the folder `main` to folder 
`app/src/main/cpp` in Android project. Implement inferrence or training process
as in the `regression` example. Update `CMakeLists.txt` by adding the source 
files from `main` to the build target, and include Eigen headers either with 
`include_directories` or `target_include_directories` clause. To add BLAS or 
OpenMP support link the target with `openblas` or `openmp` libraries.
