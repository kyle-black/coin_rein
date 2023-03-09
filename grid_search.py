import itertools
import numpy as np

class GridSearch:
    def __init__(self, agent, env, param_grid, n_episodes):
        self.agent = agent
        self.env = env
        self.param_grid = param_grid
        self.n_episodes = n_episodes
        self.results = {}

    def fit(self):
        param_values = [x for x in self.param_grid.values()]
        for params in itertools.product(*param_values):
            param_dict = {k: v for k, v in zip(self.param_grid.keys(), params)}
            self.agent.epsilon = param_dict['epsilon']
            self.agent.epsilon_decay = param_dict['epsilon_decay']
            self.agent.learning_rate = param_dict['learning_rate']
            self.agent.batch_size = param_dict['batch_size']
            self.agent.gamma = param_dict['gamma']
            scores = []
            for episode in range(self.n_episodes):
                state = self.env.reset()
                done = False
                score = 0
                while not done:
                    action = self.agent.act(state)
                    next_state, reward, done, info = self.env.step(action)
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    score += reward
                    self.agent.replay()
                scores.append(score)
            avg_score = np.mean(scores)
            self.results[str(param_dict)] = avg_score
            print(f'params: {param_dict}, avg score: {avg_score}')
        best_params = max(self.results, key=self.results.get)
        best_score = self.results[best_params]
        return eval(best_params), best_score