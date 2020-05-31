# Navigation-Bananas-Udacity-RL
Training a deep reinforcement learning agent to gather yellow bananas and avoid blue

# Submission to Project 1: Navigation for Udacity

### Introduction
This project is training a deep reinforcement learning agent to gather yellow
bananas and avoid blue bananas in a unity environment provided by udacity in 
the context of their nanodegree program on deep reinforcement learning.

### Getting Started

To run the code a unity environment is needed that can be downloaded as desribed below:

1. Download the environment from one of the links below.
   You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in this GitHub repository, in its main folder, and unzip
   (or decompress) the file. 
   
3. Dependencies:
    pip install ...
    
    matplotlib
    numpy>=1.11.0
    jupyter
    time
    cProfile
    unityagents==0.4.0
    torch==0.4.0
    ipykernel

### Project Details

For this project, an agent is trained to navigate (and collect bananas!) in 
a large, square world.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1
is provided for collecting a blue banana.  Thus, the goal of your agent is to
collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along 
with ray-based perception of objects around agent's forward direction.
Given this information, the agent has to learn how to best select actions.  
Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must
get an average score of +13 over 100 consecutive episodes.

### Description of Files

* Navigation.ipynb - 
Base file to train the agent (jupyter notebook)

* dqn_agent.py and dqn_agent_per.py - 
Classes for the deep Q-RL-agents. With epxerience replay implenented and
prioritized experience in the per-file.

* model.py - 
Class for constructing the neural net to approximate the Q function

* trained_banana_hunter.pth - 
The network weights for the trained and successful agent that achieved a score
greater 15 after 900 episodes training.

* ParameterSearch.xlsx - 
Documentation for training runs over different hyperparameters and use of
prioritized experience replay memory
