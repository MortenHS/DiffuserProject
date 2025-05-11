import argparse
import d4rl
import gym
import os
import matplotlib.pyplot as plt
import numpy as np



def visualize_env(env_name: str, output_dir="frames", stages=5, step_size=10000):
    """
    Visualize the dataset of the specified D4RL environment, combining multiple frames into one image.
    
    Args:
        env_name (str): The name of the environment to visualize.
        output_dir (str): Directory to save the rendered images.
        stages (int): Number of states to combine in one image.
        step_size (int): Interval of timesteps between stages.
    """
    env = gym.make(env_name)
    dataset = env.get_dataset()

    if 'infos/qpos' not in dataset:
        raise ValueError('Only MuJoCo-based environments can be visualized')

    qpos = dataset['infos/qpos']
    qvel = dataset['infos/qvel']
    rewards = dataset['rewards']
    actions = dataset['actions']
    
    env.reset()
    qpos[-1] = np.array([5,5])
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    total_frames = qpos.shape[0]
    
    # Use this range to iterate through the states
    for start in range(0, total_frames, stages * step_size):
        combined_frames = []

        # Collect states for t-2, t-1, t, t+1, t+2
        for stage in range(-2, 3):  # This will generate offsets -2, -1, 0, 1, 2
            t = start + (stages // 2 + stage) * step_size 

            if 0 <= t < total_frames:  # Check boundaries
                env.set_state(qpos[t], qvel[t])
                frame = env.render(mode='rgb_array', width=320, height=240)
                combined_frames.append(frame)

        # Combine frames horizontally or vertically
        if combined_frames:
            combined_image = np.concatenate(combined_frames, axis=1)  # Horizontal combination
            frame_path = os.path.join(output_dir, f"progression_{start:06d}.png")
            plt.imsave(frame_path, combined_image)

    print(f"Progression images saved to {output_dir}")


if __name__ == "__main__":
    visualize_env("maze2d-umaze-v1")
