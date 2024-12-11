import os
import csv
import json
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# Load Environment and Diffusion Model
env = datasets.load_environment(args.dataset)
diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)
diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer
policy = Policy(diffusion, dataset.normalizer)

# Storage
trajectory_data = {
    "Next_Waypoint_Method": [],
    "Actions_Method": []
}
pos_error_data = {
    "Next_Waypoint_Method": [],
    "Actions_Method": []
}

def run_method(method_name, use_waypoint_method):
    run_savepath = join(args.savepath, f'{method_name}_set_state')
    os.makedirs(run_savepath, exist_ok=True)
    observation = env.reset()
    if args.dataset == "maze2d-umaze-v1":
        env.set_state(np.array([3.03665433, 2.93015904]), np.array([0.00658355, -0.00951007]))
    elif args.dataset == "maze2d-medium-v1":
        env.set_state(np.array([2.926775177, 1.956217731]), np.array([-0.02120387, -0.09685921]))
    elif args.dataset == "maze2d-large-v1":
        env.set_state(np.array([0.94333326, 1.09938711]), np.array([0.10727024, 0.05407418]))

    target = env._target
    cond = {diffusion.horizon - 1: np.array([*target, 0, 0])}
    rollout = [observation.copy()]

    trajectory = [observation[:2].copy()]
    pos_error = []

    total_reward = 0  

    for t in range(env.max_episode_steps):
        state = env.state_vector().copy()

        if t == 0:
            cond[0] = observation
            action, samples = policy(cond, batch_size=args.batch_size)
            actions = samples.actions[0]
            sequence = samples.observations[0]

        if use_waypoint_method:
            # Next Waypoint Method
            if t < len(sequence) - 1:
                next_waypoint = sequence[t + 1]
            else:
                next_waypoint = sequence[-1].copy()
                next_waypoint[2:] = 0
            action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        else:
            # Actions Method
            if len(actions) > 1:
                action = actions[0]
                actions = actions[1:]
            else:
                action = -state[2:]

        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        rollout.append(next_observation.copy())

        trajectory.append(next_observation[:2].copy())
        pos_error.append(np.linalg.norm(next_observation[:2] - target[:2]))
        trajectory_data[method_name] = trajectory
        pos_error_data[method_name] = pos_error

        if terminal:
            break

        observation = next_observation

    # Save Rollout JSON
    json_data = {
        'score': env.get_normalized_score(total_reward),
        'step': t,
        'return': total_reward,
        'term': terminal,
        'epoch_diffusion': diffusion_experiment.epoch,
    }
    json_path = join(run_savepath, 'rollout.json')
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)


def save_plots(run_savepath, trajectory_m1, trajectory_m2, pos_error_m1, pos_error_m2, target):
    assert trajectory_m1 is not None, "trajectory_m1 is None"
    assert trajectory_m2 is not None, "trajectory_m2 is None"
    assert pos_error_m1 is not None, "pos_error_m1 is None"
    assert pos_error_m2 is not None, "pos_error_m2 is None"

    # Save Trajectory Plot
    plt.figure(figsize=(8, 6))
    traj_waypoint = np.array(trajectory_m1)
    traj_action = np.array(trajectory_m2)
    plt.plot(traj_waypoint[:, 0], traj_waypoint[:, 1], label="Separated prediction", color="blue")
    plt.plot(traj_action[:, 0], traj_action[:, 1], label="Concurrent prediction", color="red")
    plt.scatter(target[0], target[1], color="green", label="Goal", marker="X", s=100)
    plt.title("Trajectory Comparison")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid()
    traj_plot_path = join(run_savepath, 'trajectory_comp.png')
    plt.savefig(traj_plot_path)
    plt.close()

    # Plot Positional Error for both methods
    plt.figure(figsize=(8, 6))
    pos_error_waypoint = np.array(pos_error_m1)
    pos_error_action = np.array(pos_error_m2)
    plt.plot(pos_error_waypoint, label="Separated prediction", color="blue")
    plt.plot(pos_error_action, label="Concurrent prediction", color="red")
    plt.title("Positional Error Comparison")
    plt.xlabel("Time Step")
    plt.ylabel("Error (Euclidean Distance)")
    plt.legend()
    plt.grid()
    error_plot_path = join(run_savepath, 'pos_error_comp.png')
    plt.savefig(error_plot_path)
    plt.close()


# # Run Both Methods
run_method("Next_Waypoint_Method", use_waypoint_method=True)
run_method("Actions_Method", use_waypoint_method=False)

trajectory_m1 = trajectory_data["Next_Waypoint_Method"]
trajectory_m2 = trajectory_data["Actions_Method"]
pos_error_m1 = pos_error_data["Next_Waypoint_Method"]
pos_error_m2 = pos_error_data["Actions_Method"]
save_plots(args.savepath, trajectory_m1, trajectory_m2, pos_error_m1, pos_error_m2, env._target)