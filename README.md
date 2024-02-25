# Monty Hall Problem - Q-Learning Solution

This repository contains a Python implementation of a Q-learning agent designed to solve the Monty Hall problem, a famous statistical puzzle that demonstrates a counterintuitive probability principle. The project simulates the game show scenario where a contestant tries to maximize their chances of winning a car by deciding whether to stick with their initial choice of door or to switch to another door after one door with a goat behind it is revealed.

## Overview

The Monty Hall problem is based on a game show scenario where a contestant is presented with three doors, behind one of which is a car (win) and behind the others, goats (lose). After the contestant picks a door, the host, who knows what's behind the doors, opens another door, revealing a goat. The contestant then has the opportunity to stick with their original choice or switch to the other unopened door. The puzzle revolves around the best strategy to maximize the chances of winning the car.

This project uses a Q-learning reinforcement learning algorithm to learn the optimal strategy through repeated play of the Monty Hall game.

## Features

- Simulation of the Monty Hall problem environment.
- Implementation of a Q-learning agent to learn the game strategy.
- Tracking and visualization of the agent's win rate over episodes to demonstrate learning progress and strategy convergence.

### Prerequisites

Ensure you have Python 3 installed on your system. This project uses numpy and matplotlib for numerical operations and visualizations, respectively.




