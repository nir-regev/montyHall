# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 20:21:34 2024

@author: Dr. Nir Regev
"""

import numpy as np
import matplotlib.pyplot as plt

class MontyHallEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # Randomly place the car behind one of the 3 doors
        self.car_position = np.random.randint(0, 3)
        # No initial state before the contestant's choice
        return "start"

    def step(self, action):
        # Assume the contestant initially picks door 0
        initial_choice = 0
    
        # The host opens a door (1 or 2) that is not the car and not the initial choice
        # (For simplicity, this is abstracted away since it doesn't affect the probabilities directly)
        
        # If the contestant switches, they win if their initial choice wasn't the car.
        if action == 1:  # Switch
            win = 1 if self.car_position != initial_choice else 0
        else:  # Stick
            win = 1 if self.car_position == initial_choice else 0
        
        # The state doesn't actually change in this simple scenario
        return "end", win, True  # Always ends after one step

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_rate=0.95, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.99):
        self.q_table = np.zeros(2)  # Only two actions: stick or switch
        self.lr = learning_rate
        self.gamma = discount_rate
        self.epsilon = exploration_rate
        self.min_epsilon = min_exploration_rate
        self.epsilon_decay = exploration_decay

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 2)  # Explore
        else:
            return np.argmax(self.q_table)  # Exploit

    def update(self, action, reward):
        self.q_table[action] += self.lr * (reward + self.gamma * np.max(self.q_table) - self.q_table[action])
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# Training the agent
env = MontyHallEnv()
agent = QLearningAgent()

episodes = 10000
win_counts = 0
win_rates = []

for episode in range(episodes):
    state = env.reset()
    action = agent.act()
    next_state, reward, done = env.step(action)
    agent.update(action, reward)
    win_counts += reward
    win_rates.append(win_counts / (episode + 1))
    
# print(f"Winning rate: {wins / episodes}")
# print("Q-values:", agent.q_table)

# Plotting the win rates over episodes
plt.figure(figsize=(10, 6))
plt.plot(win_rates, label='Win Rate')
plt.axhline(y=2/3, color='r', linestyle='--', label='Optimal Win Rate (2/3)')
plt.xlabel('Episodes')
plt.ylabel('Win Rate')
plt.title('Win Rate of Q-Learning Agent Over Episodes')
plt.legend()
plt.show()
