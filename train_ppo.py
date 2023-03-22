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
    
    print(f"Episode {episode+1}/{1000} - Total Reward: {total_reward}")

# Save model
agent.model.save('model.h5')
print("Model saved to 'model.h5'")
