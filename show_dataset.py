import minari
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the dataset
dataset_name = "D4RL/pointmaze/large-v2"
dataset = minari.load_dataset(dataset_name)

sampled_episodes = dataset.sample_episodes(10) # Samples 10 random episodes
observations = sampled_episodes[1].observations

achieved_goal = observations['achieved_goal']
observation = observations['observation']
desired_goal = observations['desired_goal']

achieved_x = [point[0] for point in achieved_goal]
achieved_y = [point[1] for point in achieved_goal]

desired_x = [point[0] for point in desired_goal]
desired_y = [point[1] for point in desired_goal]

observation_x = [point[0] for point in observation]
observation_y = [point[1] for point in observation]

plt.figure(figsize=(8, 6))  # Set figure size for better visibility
plt.plot(achieved_x, achieved_y, marker='o', linestyle='-',linewidth=3,alpha=0.7, color='b', label='Achieved Goal')
plt.plot(desired_x, desired_y, marker='s', linestyle='--', color='r', label='Desired Goal')
# plt.plot(observation_x, observation_y, marker='*', linestyle="-.",linewidth=1,alpha=0.7, color='g',label='Observation')
plt.title("Achieved vs Desired Goals")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig("dataset_plotted.jpeg", format='jpeg', dpi=300)
plt.show()
