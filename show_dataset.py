import minari
import random
import matplotlib.pyplot as plt

# Load the dataset
dataset = minari.load_dataset("D4RL/pointmaze/large-v2")

# Access episodes
episodes = dataset.replay_buffer.episodes

# Extract trajectories (assuming observations have 'position' keys)
trajectories = []
for episode in episodes:
    trajectory = [obs["observation"][:2] for obs in episode]  # Extract x, y from observations
    trajectories.append(trajectory)

# Sample a few random trajectories
sample_trajectories = random.sample(trajectories, 5)

# Plot sampled trajectories
plt.figure(figsize=(8, 8))
for trajectory in sample_trajectories:
    trajectory = list(zip(*trajectory))  # Separate x and y
    plt.plot(trajectory[0], trajectory[1], alpha=0.7, label="Trajectory")

plt.title("Sampled PointMaze Trajectories (umaze-v2)")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid()
plt.axis("equal")  # Ensure the scale is uniform
plt.show()
