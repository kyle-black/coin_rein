class PPOAgent:
    def __init__(self, env, optimizer, gamma=0.99, clip_ratio=0.2, pi_lr=0.0003, vf_lr=0.001):
        self.env = env
        self.state_size = env.observation_space.shape
        self.action_size = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]  # Define the action bound here
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.old_actor = clone_model(self.actor)
        self.old_actor.set_weights(self.actor.get_weights())
    
    def get_action(self, state):
        state = np.reshape(state, [1, self.state_size[0]])
        prob = self.actor.predict([state, np.zeros((1, 1)), np.zeros((1, self.action_size))])[0]
        action = np.random.choice(self.action_size, size=1, p=prob[0])[0]
        return action
    


class PPOAgent:
    def __init__(self, env, optimizer, gamma=0.99, clip_ratio=0.2, pi_lr=0.0003, vf_lr=0.001):
        self.env = env
        self.state_size = env.observation_space.shape
        self.action_size = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]  # Define the action bound here
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.old_actor = clone_model(self.actor)
        self.old_actor.set_weights(self.actor.get_weights())
    
    def get_action(self, state):
        state = np.reshape(state, [1, self.state_size[0]])
        prob = self.actor.predict([state, np.zeros((1, 1)), np.zeros((1, self.action_size))])[0]
        action = np.random.choice(self.action_size, size=1, p=prob[0])[0]
        return action

    def ppo_loss(self, advantage, old_prediction):
        def loss(y_true, y_pred):
            new_prediction = y_pred
            ratio = K.exp(K.log(new_prediction + 1e-10) - K.log(old_prediction + 1e-10))
            p1 = ratio * advantage
            p2 = K.clip(ratio, min_value=1 - self.clip, max_value=1 + self.clip) * advantage
            actor_loss = -K.mean(K.minimum(p1, p2))

            critic_loss = K.mean(K.square(y_true - y_pred))

            total_loss = actor_loss + 0.5 * critic_loss

            return total_loss

        return loss
    
    def train(self, states, actions, rewards, dones):
        # Compute discounted rewards
        discounted_rewards = []
        discounted_reward = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted
