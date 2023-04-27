# 503GradProject

## Introduction
Reinforcement Learning (RL) in robotics has recently gained considerable popularity. Currently, most RL implementations use the [PyTorch](https://pytorch.org/) 
library. Google [JAX](https://github.com/google/jax) is another high-performance Python library, and its JIT compilation feature can significantly 
lower the neural network training time. 
RL training in real-time physical robots can hugely benefit from faster training time, as training with physical robots is quite expensive 
(in terms of time, effort, and cost). In this project, I compare the training time of RL implementations based on PyTorch and Jax libraries. 
This project uses a virtual environment for training the RL agents. However, the results of the experiments should be similar for RL agents trained 
using a physical robot.
