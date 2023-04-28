# 503GradProject

## Introduction
Reinforcement Learning (RL) in robotics has recently gained considerable popularity. Currently, most RL implementations use the [PyTorch](https://pytorch.org/) library. Google [JAX](https://github.com/google/jax) is another high-performance Python library, and its JIT compilation feature can significantly lower the neural network training time. RL training in real-time physical robots can hugely benefit from faster training time, as training with physical robots is quite expensive (in terms of time, effort, and cost). I am interested in RL experiments with physical robots that use image and proprioception for the state space. However, I could not find a JAX-based RL implementation that supports both image and proprioception for the state space. Furthermore, the RLAI lab already has a PyTorch-based library for training RL agents in physical robots ([ReLoD](https://github.com/rlai-lab/relod)). This project aims to create a JAX-based RL implementation that supports both image and proprioception in the state space and to test the training time performance between the RLAI's PyTorch-based implementation and the modified JAX-based implementation I created. These experiments' results will help us determine which libraries to focus on for the next experiments with physical robots in the RLAI lab. This project uses a virtual environment for training the RL agents. Although the training is conducted in a virtual environment, the results obtained should be comparable to those achieved by RL agents trained on a physical robot. 


## Setup
### Environment

The environment used in this project is the [Gym Reacher-V2](https://www.gymlibrary.dev/environments/mujoco/reacher/) environment. The task is to control a two-joint robot arm to move its end effector close to a target spawned at random positions. The environment is modified to provide image observations alongside the default proprioceptions ([source](https://github.com/rlai-lab/relod/tree/main/relod/envs/mujoco_visual_reacher)). The action space consists of two variables representing the torques applied at the two hinge joints. The reward encourages the agent to move the end effector close to the target without making large movements.  
The state space has two parts, image and proprioception. For the image, three consecutive images from the environment are stacked together, resulting in an image of the shape 125x200x9. The proprioception is a vector of size 11. 
    
![image](https://raw.githubusercontent.com/fahimfss/503GradProject/master/env.png)  
Fig: The Reacher-V2 environment   
      
     
       
### Implementation 
JAX and PyTorch implementation of the Soft Actor-Critic (SAC) RL algorithm was used in this project. Both implementations use similar neural networks to make the training time comparable.
#### JAX implementation
This project uses a modified JAX implementation of SAC found in the following two repositories: [https://github.com/henry-prior/jax-rl](https://github.com/henry-prior/jax-rl), [https://github.com/ikostrikov/jaxrl2](https://github.com/ikostrikov/jaxrl2). The main alteration I implemented was adding the support for state space based on both images and proprioception. The code can be found in the [Jax](https://github.com/fahimfss/503GradProject/tree/master/Jax) directory. To run the Jax implementation, just run the `task_mujoco.py` file using python.
#### PyTorch implementation
The PyTorch based SAC is based on the implementation found here: [https://github.com/rlai-lab/relod](https://github.com/rlai-lab/relod). In this project, I used the sequential version of SAC, where data collection and training are done in the same Python process. The code can be found in the [PyTorch](https://github.com/fahimfss/503GradProject/tree/master/PyTorch) directory. To run the PyTorch implementation, just run the `task_mujoco_visual_reacher.py` file using python.

I conducted each experiment for 25,000 steps in the Reacher environment. For each experiment, the training of the neural networks started after the first 1,000 steps.

## Results
As this project focuses on comparing the training time performance of JAX and PyTorch based implementations of image-proprioception based SAC, only those results will be discussed in this section. The complete log for individual runs can be found [here](https://github.com/fahimfss/503GradProject/tree/master/results).

I made two individual runs for JAX and two individual runs for PyTorch. Because the results for the runs were almost identical, I did not make any further runs. The results are based on the average training time for SAC neural networks with a batch size of 256. 

| Implementation | Avg training time for Run 1 | Avg training time for Run 2 | Number of training steps for each run|
| --- | --- | --- | --- |
| JAX | 0.009829s or 9.829ms | 0.009988s or 9.988ms | 24000 |
| PyTorch |  0.068393s or 68.393ms | 0.068424s or 68.424ms | 23999 | 

The result shows that, the training time for JAX based SAC implementation is a fraction of the training time for PyTorch based SAC implementation. I conclude that, using JAX based implementations for training RL agents with physical robots will significantly improve the total training time of the agent.  

## Future Works
Here are the things I want to do next:  
* Further test the JAX implementation for bug fixes and performance improvement.
* Implement an Asynchronous version of SAC using JAX
* Compare the performance of JAX based implementation with PyTorch based implementation using a physical robot.
