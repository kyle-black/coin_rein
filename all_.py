#train_ppo.py

import pandas as pd
from trading_env import TradingEnv
from ppo import PPOAgent
import numpy as np
from tensorflow.keras.optimizers import Adam

# Load data
df = pd.read_csv('btc2.csv')
data = df[['Open', 'High', 'Low', 'Price']]
data = data.replace(',', '', regex=True)
data = data.replace('K', '', regex=True)
data = data.astype(float)
data = data[['Open', 'High', 'Low', 'Price']].values

# Set up environment
env = TradingEnv(data, capital =1000)

# Define scaling factor to scale actions within action space bounds
scaling_factor = (env.action_space.high - env.action_space.low) / 2

optimizer = Adam(learning_rate=0.001)
agent = PPOAgent(env, optimizer)

# Train agent
for episode in range(1000):
    state = env.reset()
    states, actions, rewards, dones = [], [], [], []
    total_reward = 0
    done = False
    while not done:
        action = agent.get_action(state)
        action = action * scaling_factor
        action = np.clip(action, env.action_space.low, env.action_space.high)
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        state = next_state
        total_reward += reward

    # Handle empty rewards list
    if not rewards:
        rewards.append(0)

    # Train the agent
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    dones = np.array(dones)
    agent.train(states, actions, rewards, dones)
    
    print(f"Episode {episode+1}/{1000} - Total Reward: {total_reward:.2f}")

# Save model
agent.model.save('model.h5')
print("Model saved to 'model.h5'")


# ppo.py


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow import keras
import math

import tensorflow_probability as tfp
from tensorflow.keras.models import clone_model

class PPOAgent:
    def __init__(self, env, optimizer, gamma=0.99, eps=0.2, entropy_coef=0.01, vf_coef=0.5, batch_size =32):
        self.env = env
        self.state_size = env.observation_space.shape
        self.action_size = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        self.action_space = env.action_space  # Define action space attribute
        self.optimizer = optimizer
        self.gamma = gamma
        self.eps = eps
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.old_log_probs = tf.zeros((1, self.action_size))
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer.learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer.learning_rate)
        self.scaling_factor = (self.env.action_space.high - self.env.action_space.low) / 2
        self.buffer_size = 64
        self.batch_size = batch_size
        self.epochs = 5

    def build_actor(self):
        state_input = layers.Input(shape=self.state_size, name='state_input')
        x = layers.Dense(64, activation='relu')(state_input)
        x = layers.Dense(32, activation='relu')(x)
        mu_output = layers.Dense(self.action_size, activation='tanh', name='mu_output')(x)
        mu = layers.Lambda(lambda x: x * self.action_bound)(mu_output)
        std_output = layers.Dense(self.action_size, activation='softplus', name='std_output')(x)
        std = layers.Lambda(lambda x: x + 1e-5)(std_output)  # add small constant to avoid division by zero
        actor = keras.Model(inputs=state_input, outputs=[mu, std], name='actor')
        return actor

    def build_critic(self):
        state_input = layers.Input(shape=self.state_size, name='state_input')
        x = layers.Dense(64, activation='relu')(state_input)
        x = layers.Dense(32, activation='relu')(x)
        value_output = layers.Dense(1, name='value_output')(x)
        critic = keras.Model(inputs=state_input, outputs=value_output, name='critic')
        return critic

    def get_action(self, state):
        state = state.reshape(1, 30, 4)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        mu, std = self.actor(state)
        dist = tfp.distributions.Normal(mu, std)
        action = dist.sample()
        action = np.squeeze(action.numpy(), axis=0)
        action = action * self.scaling_factor
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action


    def train(self, states, actions, rewards, dones):
        # Compute discounted rewards
        discounted_rewards = []
        discounted_reward = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_rewards.insert(0, discounted_reward)

        # Normalize discounted rewards
        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        values = self.critic.predict(states).flatten()
        advantages = discounted_rewards - values

        # Pad advantages array if necessary
        num_samples = len(states)
        remainder = advantages.shape[0] % self.batch_size
        if remainder != 0:
            pad_width = ((0, self.batch_size - remainder), (0, 0))
            advantages = np.pad(advantages, pad_width, mode='constant')

        # Reshape advantages to have dimensions (num_batches, batch_size, 1)
        new_batch_size = advantages.shape[0] // self.batch_size
        advantages = np.reshape(advantages, (new_batch_size, self.batch_size, 1))

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((states, actions, advantages, discounted_rewards))
        dataset = dataset.shuffle(self.buffer_size).batch(self.batch_size)

        # Train actor and critic networks
        for epoch in range(self.epochs):
            for batch in dataset:
                states_batch, actions_batch, advantages_batch, discounted_rewards_batch = batch
                with tf.GradientTape() as tape:
                    # Compute actor loss
                    mu, std = self.actor(states_batch)
                    dist = tfp.distributions.Normal(mu, std)
                    log_probs = dist.log_prob(actions_batch)
                    ratio = tf.exp(log_probs - tf.stop_gradient(self.old_log_probs))
                    actor_loss = -tf.reduce_mean(tf.minimum(ratio * tf.expand_dims(advantages_batch, axis=-1),
                                            tf.clip_by_value(ratio, 1 - self.eps, 1 + self.eps) * tf.expand_dims(advantages_batch, axis=-1)))

                    # Compute critic loss
                    values_batch = self.critic(states_batch)
                    critic_loss = tf.reduce_mean(tf.square(values_batch - discounted_rewards_batch))

                    # Compute entropy bonus
                    entropy = tf.reduce_mean(dist.entropy())

                    # Compute total loss
                    loss = actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy

                # Update actor and critic networks
                actor_grads = tape.gradient(loss, self.actor.trainable_variables)
                critic_grads = tape.gradient(loss, self.critic.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

            # Update old policy
            self.old_log_probs = dist.log_prob(actions_batch)



            return actor_loss.numpy(), critic_loss.numpy(), entropy.numpy()



# trading_env.py

import gym
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    def __init__(self, data, capital):
        self.data = data
        self.capital = capital
        self.position = 0
        self.current_step = None
        self.steps_left = None
        self.trades = []
        self.profits = []
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(30, 4), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.steps_left = len(self.data) - 1
        self.trades = []
        self.profits = []
        return self._next_observation()

    def _next_observation(self):
        frame = np.zeros((30, 4))
        frame[0] = self.data[self.current_step]
        if self.current_step > 0:
            for i in range(1, 30):
                frame[i] = self.data[self.current_step - i]
        return frame

    def step(self, action):
        self.current_price = self.data[self.current_step][3]
        self.trades.append(self.current_price)
        self.steps_left -= 1
        self.position += action[0]

        if self.position > 1:
            self.position = 1
        elif self.position < -1:
            self.position = -1

        if self.steps_left == 0:
            profit = self.capital + self.position * self.current_price - 10000
            self.profits.append(profit)
            reward = profit / 10000
            return np.array(self._next_observation()), reward, True, {}

        if self.current_step == len(self.data) - 1:
            reward = self.capital - 10000
            self.profits.append(reward)
            return np.array(self._next_observation()), reward / 10000, True, {}

        if action[0] == 0:
            reward = 0
        else:
            profit = self.capital + self.position * self.current_price - 10000
            self.capital += self.position * self.current_price
            self.position = 0
            self.profits.append(profit)
            reward = profit / 10000

        self.current_step += 1
        obs = np.array(self._next_observation())
        done = False
        if self.steps_left == 0:
            done = True
            reward = self.profits[-1] / 10000

        return obs, reward, done, {}
