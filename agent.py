import numpy as np
from build_model import *
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from experience_replay import *

#https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/blob/master/DQN/tf2/agent.py
class Agent: 
    def __init__(self, agent_params):
        # Parameters
        self.gamma = agent_params.get("gamma")
        self.lr = agent_params.get("lr")
        self.input_dims = agent_params.get("input_dims")
        self.batch_size = agent_params.get("batch_size")
        self.replace_target_weight_counter = agent_params.get("replace")
        self.algo = agent_params.get("algo")
        self.env_name = agent_params.get("env_name")
        self.chkpt_dir = agent_params.get("chkpt_dir")

        self.learn_step_counter = 0
        self.fname = self.chkpt_dir + self.env_name + '_' + self.algo + '_'
        self.mem_size = mem_size

        # networks and replaybuffer
        self.memory = ExperienceReplayBuffer(mem_size, input_dims, n_actions)
        self.model_builder = ModelBuilder()
        self.q_value_network = self.model.builder.build_model(self.input_dims, self.n_actions, self.lr)
        self.target_q_network = self.model.builder.build_model(self.input_dims, self.n_actions, self.lr)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_experience(state, action, reward, state_, done)

    def sample_experience(self, batch_size): 
        state, action, reward, new_state, done = self.memory.sample_experience(self.batch_size)
        states = tf.convert_to_tensor(state)
        rewards = tf.convert_to_tensor(reward)
        dones = tf.convert_to_tensor(done)
        actions = tf.convert_to_tensor(action, dtype=tf.int32)
        next_states = tf.convert_to_tensor(new_state)
        return states, actions, rewards, next_states, dones

    def copy_weights_to_target_network(self): 
        if self.learn_step_counter % self.replace_target_weight_counter == 0:
            self.target_q_network.set_weights(self.q_value_network.get_weights())

    def compute_loss(self, preds, targets):
      return tf.reduce_mean(tf.square(tf.subtract(preds, targets)))

    def learn(self): 
        if self.batch_size > self.mem_size:
            return 

        self.copy_weights_to_target_network()

        states, actions, rewards, states_1, dones = self.sample_experience(self.batch_size)

        indices = tf.range(self.batch_size, dtype=tf.int32)
        action_indices = tf.stack([indices, actions], axis=1)

        with tf.GradientTape() as tape:
            q_preds = tf.gather_nd(self.q_value_network(states), indices=action_indices) 
            q_next = self.target_q_network.predict(states_1)

            max_actions = tf.math.argmax(q_next, axis=1, output_type=tf.int32)
            max_action_idx = tf.stack([indices, max_actions], axis=1)

            
            q_targets = rewards + \
                        self.gamma*tf.gather_nd(q_next, indices=max_action_idx) *\
                        (1 - dones.numpy())

            #loss = keras.losses.MSE(q_preds, q_targets)
            loss = self.compute_loss(q_preds, q_targets)

        params = self.q_value_network.trainable_variables
        grads = tape.gradient(loss, params)

        self.q_value_network.optimizer.apply_gradients(zip(grads, params))
        self.learn_step_counter += 1

        return loss

    def load_model(self): 
        self.q_value_network = keras.models.load_model(self.fname+'q_value_network.hd5', save_format="hd5")
        self.target_q_network = keras.models.load_model(self.fname+'target_q_network.hd5', save_format="hd5")
        print('[+] models loaded successfully')

    def save_model(self): 
        self.q_value_network.save(self.fname+'q_value_network.hd5', save_format="hd5")
        self.target_q_network.save(self.fname+'target_q_network.hd5', save_format="hd5")
        print('[+] models saved successfully')
