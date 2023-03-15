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
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(120,))
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
    '''
    def _get_reward(self, action):
        previous_price = self.data[self.current_step - 1]


        #current_price = self.data[self.current_step]
        if self.current_step +1 >=len(self.data):
            current_price = self.data[self.current_step]
        
        else:
            current_price =self.data[self.current_step +1]        
        
        if action == 1:  # Buy
            reward = current_price - previous_price
        else:  # Hold or sell
            reward = 0
        return reward
    '''
    def _get_reward(self, action):
        try:
            previous_price = self.data[self.current_step - 1]
            current_price = self.data[self.current_step]
            if self.current_step + 1 >= len(self.data):
                current_price = current_price
            else:
                current_price = self.data[self.current_step + 1]
            if action == 1:  # Buy
                reward = current_price - previous_price
            else:  # Hold or sell
                reward = 0
        except IndexError:
            reward = 0
        return reward
    
    '''
    def _get_reward(self, action):
        current_price = self.data[self.current_step]
        if action == 0:  # Buy
            self.bought_price = current_price
            return 0
        elif action == 1 and self.bought_price is not None:  # Sell
            diff = current_price - self.bought_price
            self.bought_price = None
            return diff
        else:  # Hold
            return 0
    '''
