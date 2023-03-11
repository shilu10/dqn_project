import gymnasium as gym
import time
import os 
from utils import * 
from agent import * 
from trainer import *
import numpy as np

env = make_env("LunarLander-v2", "videos/", 50)
action_space = [_ for _ in range(env.action_space.n)]

episodic_rewards_filename = 'array_files/episodic_reward.npy'
epsilon_history_filename = 'array_files/epsilon_history.npy'
cum_avg_reward_filename = 'array_files/cum_avg_rewards.npy'
losses_filename = 'array_files/losses.npy'

agent_params = {
    "gamma": 0.98, 
    "lr": 0.0005, 
    "input_dims": env.observation_space.shape,
    "mem_size" : 15000,
    "batch_size" : 32,
    "replace" : 800,
    "algo" : "DQN",
    "env_name" : "lunarlander",
    "n_actions" : len(action_space),
    "chkpt_dir": "tmp/dqn/"
}

trainer_params = {
    "noe": 500, 
    "max_steps": 1000,
    "max_eps": 1,
    "min_eps": 0.02,
    "eps_decay_rate": 1e-2,
    "eps": 1,
    "action_space": action_space,
    "is_tg": True,
    "tg_bot_freq_epi": 10
}

if __name__ == "__main__": 
    manage_memory()
    agent = Agent(agent_params)
    trainer = Trainer(agent, env, trainer_params)

    trainer.train_rl_model()
    
    # Loading Numpy array from npy files
    episode_rewards = np.load(episodic_rewards_filename, mmap_mode="r")
    epsilon_history = np.load(epsilon_history_filename, mmap_mode="r")
    avg_rewards = np.load(cum_avg_reward_filename, mmap_mode="r")
    losses = np.load(losses_filename, mmap_mode="r")
    
    #print(episode_rewards)
   # plot_learning_curve(10, episode_rewards, epsilon_history, "plot_file")

   # eval_model(env, "keras model", "videos/", fps=10)
