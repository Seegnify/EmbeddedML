# Seegnify

[Seegnify](https://www.seegnify.org/) is an open source federated machine learning framework that runs anywhere. It was created with the following goals in mind:

  * High performance code for easy research and development
  * Deep learning neural networks based on dynamic graph
  * Automatic distributed learning on variety of platforms

## More About Seegnify

<!-- toc -->

- [Research Areas](#research-areas)
  - [Federated Learning](#federated-learning)
- [Project Structure](#project-structure)
  - [Main Components](#main-components)
  - [Dependencies](#dependencies)
- [Installation](#installation)
  - [From Source](#from-source)
  - [Unit Test](#unit-test)
  - [Full Test](#full-test)
- [Examples](#examples)
- [Training](#training)
- [License](#license)

<!-- tocstop -->

## Research Areas

### Federated Learning

Federated Learning is a type of machine learning that occurs on multiple 
decentralized nodes with local data samples. This type of learning is useful in 
cases where data cannot be exchanged or when learning requires different data 
samples on each node. This paradigm provides strong data privacy protection.

Since Federated Learning is by definition distributed, it is also an 
effective way of implementing a parallel training for Reinforcement Learning 
where RL environment is not differentiable. RL with Seegnify is easy. A 
developer simply writes a model and a training routine for a single core and 
the framework runs it in a distributed mode automatically.

## Project Structure

### Main Components

| Component | Description                                        |
| --------- | -------------------------------------------------- |
| bin       | Build and run-time control scripts for convenience |
| main      | Distributed automatic differentiation library      |
| rl        | Reinforcement-Learning environment                 |
| examples  | Neural Network models and training samples         |

### Dependencies

| Component | Eigen | Protobuf | POCO | ImageMagick | SndFile* | Qt* |
| --------- | ----- | -------- | ---- | ----------- | -------- | --- |
| bin       | no    | no       | no   | no          | no       | no  |
| main      | yes   | yes      | yes  | no          | yes      | yes |
| rl        | no    | no       | no   | no          | no       | yes |
| examples  | no    | no       | no   | yes         | no       | yes |

(*) A work is beeing done to remove Qt and SndFile dependencies.

## Installation

### From Source

The binaries can be compiled using the provided CMake wrapper script:

```bash
./bin/build.sh
```

### Unit test

All unit tests can be executed by runing:

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

| Example    | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| MNIST      | Model definition, training and validation on MNIST dataset   |
| CIFAR10    | Model definition, training and validation on CIFAR10 dataset |
| MINST RL   | Reinforcement learning with attention on MNIST dataset       |
| CIFAR10 RL | Reinforcement learning with attention on CIFAR10 dataset     |

## Training

Training a model requires implementing the `Training` interface and 
creating the model definition. Complete reference implementations can be 
found in the folder `examples` as well as in the full test. Once 
implemented and compiled, start master process and then worker process.

Start the master process on port 2020 and store the model in my.graph:

```bash
./build/seegnify-training master my.graph 2020
```

Start the worker process with your compiled model and connect to master:

```bash
./build/seegnify-training worker 127.0.0.1 2020 ./build/libseegnify-fulltest.so
```

Workers can be started on multiple nodes. A worker by default utilizes all cores and assignes one traning instance to each core.

Stop the training by sending SIGINT (Ctrl-C) signal to the master process.

## License

Seegnify is licensed under the Apache License, Version 2.0, as found in the [LICENSE](LICENSE) file.
