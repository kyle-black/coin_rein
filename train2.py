import argparse
import pandas as pd
from trading_env import TradingEnv
from trading_agent import TradingAgent
from grid_search import GridSearch
import os







def main(args):
    # Load data
    df = pd.read_csv('btc2.csv')
    data = df[['Open', 'High', 'Low', 'Price']]

    data = data.replace(',', '',regex=True)
    data = data.replace('K','', regex =True)

    #data.astype(float)
    
    #data = data[['Open', 'High', 'Low', 'Price']].values
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
    #agent = TradingAgent(state_size, action_size, model_params, observation_space=env.observation_space, action_space=env.action_space)
    agent = TradingAgent(state_size, action_size, model_params, observation_space=env.observation_space, action_space=env.action_space)
    print(agent.model.summary())
    # Train agent
    
    # Save model
   # model_dir = args.model_dir
   # os.makedirs(model_dir, exist_ok=True)
   # model_path = os.path.join(model_dir, 'model.h5')
   # agent.model.save(model_path)
   # print(f'Model saved to {model_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
   # parser.add_argument('--data_path', type=str, required=True,
    #                    help='Path to CSV file containing stock data')
   # parser.add_argument('--model_dir', type=str, required=True,
    #                    help='Directory to save trained model')
    parser.add_argument('--window_size', type=int, default=30,
                        help='Size of the window used for each observation')
    parser.add_argument('--episodes', type=int, default=100,
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
    args = parser.parse_args()

    main(args)

