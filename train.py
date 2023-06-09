from npy_append_array import NpyAppendArray
import numpy as np
from writer import * 
from telegram_bot import * 
from 

class Trainer:   
    def __init__(self, agent, env, trainer_params): 
        self.agent = agent
        self.env = env 
        self.noe = trainer_params.get("noe")
        self.max_steps = trainer_params.get("max_steps")
        self.eps = trainer_params.get("eps")
        self.min_eps = trainer_params.get("min_eps")
        self.max_eps = trainer_params.get("max_eps")
        self.eps_decay_rate = trainer_params.get("eps_decay_rate")
        self.action_space = trainer_params.get("action_space")
        self.is_tg = trainer_params.get("is_tg")
        self.tg_bot_freq_epi = trainer_params.get("tg_bot_freq_epi")

        self.writer = Writer("model_training_results.txt")

        self.episodic_rewards_filename = 'array_files/episodic_reward.npy'
        self.epsilon_history_filename = 'array_files/epsilon_history.npy'
        self.cum_avg_reward_filename = 'array_files/cum_avg_rewards.npy'
        self.losses_filename = 'array_files/losses.npy'

    def train_rl_model(self): 
        episode_rewards = []
        epsilon_history = []
        avg_rewards = []
    #    losses = []
        best_reward = float("-inf")

        for episode in range(self.noe): 
            n_steps = 0 
            episodic_loss = 0
            state = self.env.reset()
        #  self.env.start_video_recorder()
            reward = 0 

            for step in range(self.max_steps): 

                if type(state) == tuple: 
                    state = state[0].astype("float32")
                state = state.astype("float32")
             #   print(state, state.shape, "weights")
              #  print()

                action = epsilon_greedy_policy(
                      self.eps, 
                      state,
                      agent.q_value_network,
                      self.action_space
                  )

                next_info = self.env.step(action)
                next_state, reward_prob, terminated, truncated, _ = next_info
                reward += reward_prob

                self.agent.store_transition(state, action, reward_prob, next_state, terminated or truncated)
                self.agent.learn()
             #   episodic_loss += loss 

                state = next_state
                n_steps += 1 
             #   self.eps = self.eps - self.eps_decay_rate if self.eps > self.min_eps else self.min_eps
                if terminated or truncated: 
                 # self.env.close()
                    break

            epsilon_history.append(self.eps)
            episode_rewards.append(reward)
         # losses.append(episodic_loss)
            avg_reward = np.mean(episode_rewards)
            avg_rewards.append(avg_reward)

            result = f"Episode: {episode}, epsilon: {self.eps}, Steps: {n_steps}, reward: {reward}, best reward: {best_reward}, avg rew last 50 epi: {avg_reward}"
            self.writer.write_to_file(result)
            print(result)

            if reward > best_reward: 
                best_reward = reward
                agent.save_model()
            
            self.eps = self.eps - self.eps_decay_rate if self.eps > self.min_eps else self.min_eps
          # Telegram bot
            if self.is_tg and episode % self.tg_bot_freq_epi == 0: 
                info_msg(episode+1, self.noe, reward, best_reward, "None")
                

          # used for better memory management
            if episode !=0 and episode % 10 == 0: 
                with NpyAppendArray(self.episodic_rewards_filename) as npaa: 
                    npaa.append(np.array(episode_rewards))
                    episode_rewards = []

                with NpyAppendArray(self.epsilon_history_filename) as npaa: 
                    npaa.append(np.array(epsilon_history))
                    epsilon_history = []

                with NpyAppendArray(self.cum_avg_reward_filename) as npaa: 
                    npaa.append(np.array(avg_rewards))
                    avg_rewards = []

               # with NpyAppendArray(self.losses_filename) as npaa: 
                 # npaa.append(np.array(losses))
                #  losses = []

