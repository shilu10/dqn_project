import numpy as np 

class ExperienceReplayBuffer: 
    def __init__(self, max_memory, input_shape, n_actions): 
        self.mem_size = max_memory
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_experience(self, state, action, reward, next_state, done): 
        index = self.mem_counter % self.mem_size 

        self.state_memory[index] = state[0]
        self.new_state_memory[index] = next_state[0]
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_counter += 1

    def sample_experience(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch_index = np.random.choice(max_mem, batch_size, replace=True)

        states = self.state_memory[batch_index]
        next_states = self.new_state_memory[batch_index]
        rewards = self.reward_memory[batch_index]
        actions = self.action_memory[batch_index]
        terminal = self.terminal_memory[batch_index]

        return states, actions, rewards, next_states, terminal
