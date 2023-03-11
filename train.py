from npy_append_array import NpyAppendArray
import numpy as np
from writer import *
from bot import *

class Trainer:   
  def __init__(self, agent, env, trainer_params): 
    self.agent = agent
    self.env = env 
    self.noe = trainer_params.get("noe")
    self.max_steps = trainer_params.get("max_steps")
    self.eps = trainer_params.get("eps")
    self.min_eps = trainer_params.get("min_eps")
    self.max_eps = trainer_params.get("max_eps")
    self.eps_decay = trainer_params.get("eps_decay")
    self.action_space = trainer_params.get("action_space")
    self.is_tg = trainer_params.get("is_tg")
    self.tg_bot_freq_epi = trainer_params.get("tg_bot_freq_epi")

    self.writer = Writer("model_training_results.txt")

    self.episodic_rewards_filename = 'array_files/episodic_reward.npy'
    self.epsilon_history_filename = 'array_files/epsilon_history.npy'
    self.cum_avg_reward_filename = 'array_files/cum_avg_rewards.npy'
    self.losses_filename = 'array_files/losses.npy'

  def train_rl_model(): 
    episode_rewards = []
    epsilon_history = []
    avg_rewards = []
    losses = []
    best_reward = float("-inf")

    for episode in range(self.noe): 
      n_steps = 0 
      episodic_loss = 0
      state = self.env.reset()
      reward = 0 

      for step in range(self.max_steps): 

        if type(state) == tuple: 
          state = state[0].astype("float32")
        state = state.astype("float32")
              
        action = epsilon_greedy_policy(
              self.eps, 
              state,
              agent.q_value_network,
              self.action_space
          )

        next_info = self.env.step(action)
        next_state, reward_prob, terminated, truncated, _ = next_info
        reward += reward_prob

        self.agent.store_transition(state, action, reward_prob, next_state, done)
        loss = self.agent.learn()
        episodic_loss += loss 

        state = next_state
        n_steps += 1 
        if terminated or truncated: 
          break
      
      self.eps = self.eps - self.eps_decay if self.eps > self.min_eps else self.min_eps

      epsilon_history.append(self.eps)
      episode_rewards.append(reward)
      losses.append(episodic_loss)
      avg_reward = np.mean(episode_rewards)
      avg_rewards.append(avg_reward)
      
      result = f"Episode: {episode}, epsilon: {epsilon}, Steps: {n_steps}, reward: {reward}, best reward: {best_reward}, avg rew last 50 epi: {avg_reward}"
      self.writer.write_to_file(result)
      print(result)

      if reward > best_reward: 
        best_reward = reward
        agent.save_model()

      # Telegram bot
      if self.is_tg and episode % self.tg_bot_freq_epi == 0: 
        tg.info_msg(episode+1, self.noe, reward, best_reward, loss)


      # used for better memory management
      if episode % 50 == 0: 
        with NpyAppendArray(self.episodic_rewards_filename) as npaa: 
          npaa.append(episode_rewards)
          episode_rewards = []

        with NpyAppendArray(self.epsilon_history_filename) as npaa: 
          npaa.append(epsilon_history)
          epsilon_history = []

        with NpyAppendArray(self.cum_avg_reward_filename) as npaa: 
          npaa.append(avg_rewards)
          avg_rewards = []

        with NpyAppendArray(self.losses) as npaa: 
          npaa.append(losses)
          losses = []

