"""https://github.com/vsaveris/lunar-lander-DQN"""
## The description of the task
Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.

Source: https://gym.openai.com/envs/LunarLander-v2/

## Deep Q-Learning
My implementation is inspired by the Deep Q-Learning algorithm as described in reference [2]. The input to my Deep Q-Learner are the observations of the Lunar Lander environment. The Deep Neural Network I used, is implemented in Keras (using Tensor Flow as backend). In short, the Deep Q-Learning algorithm selects actions according an ε-greedy policy. Each experience tuple <s, a, r, s’> is stored in a Replay Memory structure. On each algorithm iteration, a random sample of these stored memories (minibatch) is selected and Q-Learning updates are applied to these samples. The detailed algorithm and the advantages of this approach are described in detail in reference [2].

## Deep Q-Network
My implementation is inspired by the Deep Q-Learning algorithm as described in reference [2]. The input to my Deep Q-Learner are the observations of the Lunar Lander environment. The Deep Neural Network I used, is implemented in Keras (using Tensor Flow as backend). In short, the Deep Q-Learning algorithm selects actions according an ε-greedy policy. Each experience tuple <s, a, r, s’> is stored in a Replay Memory structure. On each algorithm iteration, a random sample of these stored memories (minibatch) is selected and Q-Learning updates are applied to these samples. The detailed algorithm and the advantages of this approach are described in detail in reference [2].

## Setup for running in local environment
Install python 3.7, if not already installed, to install in debian distro use,  

    $ sudo apt install python3.7
Install git, if not ready installed, to install in debian distro use,
    
    $ sudo apt install git 

Open terminal and execute the below commands, 

    $ git clone https://github.com/shilu10/dqn_project.git
    $ cd dqn_project/ 
    $ rm array_files/* 
    $ rm model_training_results.txt
    $ rm -r videos/ 
    $ sudo apt install python3-pip 
    $ python3 -m venv dqn_project_env/
    $ source dqn_project_env/bin/activate/
    $ pip3 install -r requirements.txt 

Now, you are ready to go and run this project refer the below sections to understand the project structure to train and test the Reinforcement Learning Model.

## Components
The implementation of the project is in python and it is included in the below files:
    lunarLander.py : Agent for landing successfully the 'Lunar Lander' which is implemented in OpenAI gym (reference [1]).

    Files:

    agent.py                 : 
    network.py               : 
    experience_replay.py     : 
    telegram_bot.py          :
    train.py                 :
    eval.py                  : 
    pong_env                 : 
    lunarlander_env.py       :
    build_model.py           : 
    utils.py                 : 
    writer.py                : 


## System Requirements (Dependencies)
The script has been developed and verified in a Python 3.7 environment. Installation details of python, can be found in the following link: Python Installation

The script imports the below packages:

    matplotlib, version 2.2.2
    numpy, version 1.14.4
    pandas, version 0.23.0
    gymnasium, version 0.10.8
    Keras, version 2.2.4
    tensorflow, version 2.11.0
Need not install all of this packages, if you followed the instruction to setting up this project.


## Executing the project (Usage)
The project can be executed as explained below:
    
    $python lunarLander.py -h
    Using TensorFlow backend.
    usage: lunarLander.py [-h] [-v {0,1,2}] -e {train,test} [-a A]

    Lunar Lander with DQN

    optional arguments:
    -h, --help       show this help message and exit
    -v {0,1,2}       verbose level (0: None, 1: INFO, 2: DEBUG)
    -e {train,test}  execute (train, test)
    -a A             trained agent file

## Training the agent
The training of the agent can be started by the following script execution:

    $python lunarLander.py -e train

    Statistics: episode = 00001, steps =   75, total_reward =  -115.704, 100_episodes_average_reward =  -115.704
    Statistics: episode = 00002, steps =   92, total_reward =  -204.172, 100_episodes_average_reward =  -159.938
    Statistics: episode = 00003, steps =  110, total_reward =  -484.729, 100_episodes_average_reward =  -268.202
    ....

The training ends when the default convergence criteria are met (he average total reward over 100 consecutive runs is at least 200). The trained agent is stored in the ./trained_model/DQN_Trained.h5 for future use.

#### Total reward per training episode, till convergence



## Testing a trained agent
By loading the stored trained agent (file ./trained_model/DQN_Trained.h5), we can test its performance by the following script execution:
    $python lunarLander.py -e test -a .\trained_model\DQN_Trained.h5

    Statistics: episode = 00001, steps =  175, total_reward =   264.139, 1_episodes_average_reward =   264.139

The rendering feature of the OpenAI Gym is set to True for this step. Below is a succesful landing performed by the trained agent:

## What's New 
Implementation of Double DQN, Double Dueling DQN, and using the PER (Priortized Experience Replay).
## References
1. OpenAI Gym, arXiv:1606.01540v1 [cs.LG].
2. Playing Atari with Deep Reinforcement.
3. Learning, arXiv:1312.5602v1 [cs.LG].
https://keras.io