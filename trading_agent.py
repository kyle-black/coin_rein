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


        # Debugging print statements
        print(f"rewards shape before: {rewards.shape}")
        rewards = rewards.reshape(-1)
        print(f"rewards shape after: {rewards.shape}")

    # Compute new targets
        new_targets = self.model.predict(states)
        for i in range(self.batch_size):
            if dones[i]:
                new_targets[i, actions[i]] = rewards[i].reshape((1,))
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

