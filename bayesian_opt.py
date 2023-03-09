from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Define the function to optimize
@use_named_args([
    Real(0.0001, 0.1, name='learning_rate'),
    Integer(16, 128, name='batch_size'),
    Real(0.1, 1.0, name='discount_factor'),
])
def trading_agent_score(learning_rate, batch_size, discount_factor):
    # Train and evaluate the trading agent with the given hyperparameters
    agent = TradingAgent(learning_rate=learning_rate, batch_size=batch_size, discount_factor=discount_factor)
    score = evaluate_agent(agent)
    return -score  # Minimize the negative score

# Define the search space and initialize the optimizer
search_space = [
    Real(0.0001, 0.1, name='learning_rate'),
    Integer(16, 128, name='batch_size'),
    Real(0.1, 1.0, name='discount_factor'),
]
optimizer = gp_minimize(trading_agent_score, search_space, n_calls=10)

# Print the best set of hyperparameters found by the optimizer
print(f"Best hyperparameters: {optimizer.x}")
