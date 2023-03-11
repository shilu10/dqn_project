import gym
import time
import flappy_bird_gym
from gym import wrappers
from agent import Agent 
from utils import * 
from eval import *
from train import Trainer

env = make_env("PongNoFrameskip-v4", "videos/", 50)
action_space = [_ for _ in range(ENV.action_space.n)]

agent_params = {
    "gamma": 0.99, 
    "lr": 5e-5, 
    "input_dims": ,
    "mem_size" = 15000,
    "batch_size" = 32,
    "replace" = 1000,
    "algo" = "DQN",
    "env_name" = "pong-v5",
    "n_actions" = len(action_space)
}

trainer_params = {
    "noe": 1000, 
    "max_steps": 10000,
    "max_eps": 1,
    "min_eps": 0.02,
    "eps_decay_rate": 10 ** 5,
    "eps": 1,
    "action_space": action_space,
    "is_tg": True,
    "tg_bot_freq_epi": 20
}

if __name__ == "__main__": 
    manage_memory()
    agent = Agent(agent_params)
    trainer = Trainer(agent, ENV, trainer_params)

    trainer.train_rl_model()
    
    # Loading Numpy array from npy files
    episode_rewards = np.load(self.episodic_rewards_filename, mmap_mode="r")
    epsilon_history = np.load(self.epsilon_history_filename, mmap_mode="r")
    avg_rewards = np.load(self.cum_avg_reward_filename, mmap_mode="r")
    losses = np.load(self.losses, mmap_mode="r")

    plot_learning_curve(10, episode_rewards, epsilon_history, "plot_file")

    eval_model(ENV, "keras model", "videos/", fps=10)

