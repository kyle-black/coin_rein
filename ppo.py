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

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((states, actions, advantages[:, np.newaxis], discounted_rewards))
        dataset = dataset.shuffle(self.buffer_size).batch(self.batch_size)

        # Train actor and critic networks
        for epoch in range(self.epochs):
            for batch in dataset:
                states_batch, actions_batch, advantages_batch, discounted_rewards_batch = batch
             # Compute actor loss and update actor network
        with tf.GradientTape() as actor_tape:
            mu, std = self.actor(states_batch)
            dist = tfp.distributions.Normal(mu, std)
            log_probs = dist.log_prob(actions_batch)
            ratio = tf.exp(log_probs - tf.stop_gradient(self.old_log_probs))
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages_batch,
                                        tf.clip_by_value(ratio, 1 - self.eps, 1 + self.eps) * advantages_batch))
            actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Compute critic loss and update critic network
        with tf.GradientTape() as critic_tape:
            values_batch = self.critic(states_batch)
            critic_loss = tf.reduce_mean(tf.square(tf.squeeze(values_batch) - discounted_rewards_batch))
            critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Compute entropy bonus
        entropy = tf.reduce_mean(dist.entropy())

        # Compute total loss
        loss = actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy
  

            # Update old policy
        self.old_log_probs = dist.log_prob(actions_batch)

        return actor_loss.numpy(), critic_loss.numpy(), entropy.numpy()

