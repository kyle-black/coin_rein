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
    early_stop_patience = 3
    no_improvement_count = 0
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
            print('episode:', episode)
            print('total reward:', total_reward)

            # Baseline model
            baseline_action = 1  # Buy and hold
            baseline_next_state, baseline_reward, _, _ = env.step(baseline_action)
            baseline_reward += baseline_reward
            

    # Update the best score and check for early stopping
        if np.max(total_reward) > best_score:
            best_score = np.max(total_reward)
            no_improvement_count = 0
            agent.model.save('best_model.h5')
            print(f"Episode {episode + 1}/{args.episodes} - New Best Score: {best_score:.2f}")
        else:
            no_improvement_count += 1
            print(f"Episode {episode + 1}/{args.episodes} - Total Reward: {total_reward:.2f}")
        if no_improvement_count >= early_stop_patience:
            print(f"No improvement for {early_stop_patience} episodes. Stopping training.")
            break

    # Load the best model and save it as the final model
    best_model_path = 'best_model.h5'
    final_model_path = 'final_model.h5'
    best_model = tf.keras.models.load_model(best_model_path)
    best_model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")


    # Evaluate baseline model
    print(f"Baseline Score: {baseline_reward:.2f}")


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

