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
