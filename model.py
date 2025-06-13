import numpy as np
import random

class QLearningChatbot:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state_idx):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.q_table.shape[1] - 1)
        else:
            return np.argmax(self.q_table[state_idx])

    def update_q_value(self, state_idx, action_idx, reward):
        best_next_action = np.argmax(self.q_table[state_idx])
        self.q_table[state_idx, action_idx] += self.alpha * (
            reward + self.gamma * self.q_table[state_idx, best_next_action]
            - self.q_table[state_idx, action_idx]
        )

    def get_best_action(self, state_idx):
        return np.argmax(self.q_table[state_idx])
