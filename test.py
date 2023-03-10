# train.py


import argparse
import pandas as pd
from trading_env import TradingEnv
from trading_agent import TradingAgent
import numpy as np
import os
import tensorflow as tf


def main(args):
    # Load data

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f"GPU found: {gpu}")
    else: print('GPU Not found')


    df = pd.read_csv('btc2.csv')
    data = df[['Open', 'High', 'Low', 'Price']]
    data = data.replace(',', '', regex=True)
    data = data.replace('K', '', regex=True)
    data = data.astype(float)
    data = data[['Open', 'High', 'Low', 'Price']].values

    # Set up environment
    env = TradingEnv(data, window_size=args.window_size)

    # Set up agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model_params = {
        'gamma': args.gamma,
        'epsilon': args.epsilon,
        'epsilon_min': args.epsilon_min,
        'epsilon_decay': args.epsilon_decay,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size
    }
    agent = TradingAgent(state_size, action_size, model_params,
                          observation_space=env.observation_space, action_space=env.action_space)
    print(agent.model.summary())

    # Train agent
    best_score = -np.inf
    for episode in range(args.episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward

            print('reward:', reward)
            print('total_reward:',total_reward)
            

        # Update the best score
        if total_reward.any() > best_score:
            best_score = total_reward

        #print(f"Episode {episode + 1}/{args.episodes} - Total Reward: {total_reward:.2f}")
        print('episode', episode)
        print('total_reward', total_reward)
    # Save model
    #model_dir = args.model_dir

    #os.makedirs(model_dir, exist_ok=True)
    #model_path = os.path.join(model_dir, 'model.h5')
    agent.model.save('model.h5')
    print(f"Model saved to {model_path}")

    # Output the best score
    print(f"Best Score: {best_score:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=30,
                        help='Size of the window used for each observation')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of training episodes to run')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Discount factor for future rewards')
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='Starting value of exploration rate')
    parser.add_argument('--epsilon_min', type=float, default=0.01,
                        help='Minimum value of exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                        help='Exponential decay rate for exploration rate')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    #parser.add_argument('--model_dir', type=str, required=True,
     #                   help='Directory to save trained model')
    args = parser.parse_args()

    main(args)

# trading_agent.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

class TradingAgent:
    def __init__(self, state_size, action_size, model_params, observation_space, action_space):
        
        self.state_size = state_size
        self.action_size = action_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.model_params = model_params
        self.model = self._build_model(state_size, action_size, model_params)
        self.memory = []
        self.gamma = model_params.get('gamma', 0.95)
        self.epsilon = model_params.get('epsilon', 1.0)
        self.epsilon_min= model_params.get('epsilon_min', 0.01)
        self.epsilon_decay = model_params.get('epsilon_decay', 0.995)
        self.learning_rate = model_params.get('learning_rate', 0.001)
        self.batch_size = model_params.get('batch_size', 32)

    def _build_model(self, state_size, action_size, model_params):
        model = Sequential()
        model.add(Dense(256, input_shape=self.observation_space.shape, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.model_params['learning_rate']))
#        model.summary()
        return model
        

    def act(self, state):
        state = state.reshape(1, -1)  # Reshape to correct shape
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state)
        return int(np.argmax(q_values[0]))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = np.array(random.sample(self.memory, self.batch_size), dtype=object)
        states = np.stack(minibatch[:, 0]).reshape((self.batch_size, self.state_size))
        actions = minibatch[:, 1].astype(int)
        rewards = minibatch[:, 2]
        next_states = np.stack(minibatch[:, 3]).reshape((self.batch_size, self.state_size))
        dones = minibatch[:, 4].astype(bool)

    # Compute new targets
        new_targets = self.model.predict(states)
        for i in range(self.batch_size):
            if dones[i]:
                new_targets[i, actions[i]] = rewards[i]
                print(new_targets)
            else:
                next_q_values = self.model.predict(next_states[i].reshape(1, -1))
                target_q_values = rewards[i] + self.gamma * np.max(next_q_values)
                print('target Q values :', target_q_values)
                
                if isinstance(target_q_values, np.ndarray):
                    new_targets[i, actions[i]] = target_q_values[0]
                else:
                    new_targets[i, actions[i]] = target_q_values


                

    # Train model
        self.model.fit(states, new_targets, epochs=1, verbose=0)

    # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# train_env.py


import numpy as np
import gym
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, data, window_size=30):
        self.data = np.array(data)
        self.num_features = self.data.shape[1]
        self.window_size = window_size
        self.current_step = self.window_size
        self.max_step = len(self.data) - 1
        self.action_space = spaces.Discrete(2)
        #self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_features * self.window_size,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(120,))
        print(f"Observation space: {self.observation_space}")
    def _get_observation(self):
        observation = self.data[self.current_step - self.window_size : self.current_step]
        observation = observation.flatten()
        return observation

    def step(self, action):
        assert self.action_space.contains(action)
        reward = self._get_reward(action)
        done = self.current_step == self.max_step
        self.current_step += 1
        next_observation = self._get_observation()
        return next_observation, reward, done, {}

    def reset(self):
        self.current_step = self.window_size
        return self._get_observation()

    def _get_reward(self, action):
        previous_price = self.data[self.current_step - 1]
        current_price = self.data[self.current_step]
        if action == 1:  # Buy
            reward = current_price - previous_price
        else:  # Hold or sell
            reward = 0
        return reward



