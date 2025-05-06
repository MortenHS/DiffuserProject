import os
import csv
import json
import torch
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

def get_euclid_storage():
    trajectory_data = {
        "Diffusion": [],
        "CFM": []
    }
    pos_error_data = {
        "Diffusion": [],
        "CFM": []
    }
    return trajectory_data, pos_error_data


def compare_euclid_pos_error(method_name):
    # Storage:
    trajectory_data, pos_error_data = get_euclid_storage()

    if method_name == "CFM":
        class Parser(utils.Parser):
            dataset: str = 'maze2d-umaze-v1'
            config: str = 'config.maze2d_cfm'

    if method_name == "Diffusion":
        class Parser(utils.Parser):
            dataset: str = 'maze2d-umaze-v1'
            config: str = 'config.maze2d'

    args = Parser().parse_args('plan')
    
    env = datasets.load_environment(args.dataset)

    diffusion_exp = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)
    print(f"Loading diffusion from: {join(args.logbase, args.dataset, args.diffusion_loadpath)}")

    diffusion = diffusion_exp.ema
    dataset = diffusion_exp.dataset
    renderer = diffusion_exp.renderer

    policy = Policy(diffusion, dataset.normalizer)
    observation = env.reset()

    if args.conditional:
        print('Resetting target')
        env.set_target()

    run_savepath = join(args.savepath, f'{method_name}_set_state')
    os.makedirs(run_savepath, exist_ok=True)
    
    if args.dataset == "maze2d-umaze-v1":
        env.set_state(np.array([3.03665433, 2.93015904]), np.array([0.00658355, -0.00951007]))
    elif args.dataset == "maze2d-medium-v1":
        env.set_state(np.array([2.926775177, 1.956217731]), np.array([-0.02120387, -0.09685921]))
    elif args.dataset == "maze2d-large-v1":
        env.set_state(np.array([0.94333326, 1.09938711]), np.array([0.10727024, 0.05407418]))

    target = env._target
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0])
    }
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

        if t < len(sequence) - 1:
            next_waypoint = sequence[t + 1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        rollout.append(next_observation.copy())

        if 'maze2d' in args.dataset:
            xy = next_observation[:2]
            goal = env.unwrapped._target
            print(
                f'maze | pos: {xy} | goal: {goal}'
            )

        ## update rollout observations
        rollout.append(next_observation.copy())

        if t % args.vis_freq == 0 or terminal:
            fullpath = join(args.savepath, f'{t}.png')

            if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)

            renderer.composite(join(args.savepath, f'rollout.png'), np.array(rollout)[None], ncol=1)

        if terminal:
            break

        observation = next_observation

    trajectory.append(next_observation[:2].copy())
    pos_error.append(np.linalg.norm(next_observation[:2] - target[:2]))
    trajectory_data[method_name] = trajectory
    pos_error_data[method_name] = pos_error
   
    # Save Rollout JSON
    json_data = {
        'score': score,
        'step': t,
        'return': total_reward,
        'term': terminal,
        'epoch_diffusion': diffusion_exp.epoch,
    }
    json_path = join(run_savepath, 'rollout.json')
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)

    return trajectory_data, pos_error_data

def save_plots(run_savepath, trajectory_m1, trajectory_m2, pos_error_m1, pos_error_m2):
    assert trajectory_m1 is not None, "trajectory_m1 is None"
    assert trajectory_m2 is not None, "trajectory_m2 is None"
    assert pos_error_m1 is not None, "pos_error_m1 is None"
    assert pos_error_m2 is not None, "pos_error_m2 is None"

    # Save Trajectory Plot
    plt.figure(figsize=(8, 6))
    traj_diff = np.array(trajectory_m1)
    traj_cfm = np.array(trajectory_m2)
    plt.plot(traj_diff[:, 0], traj_diff[:, 1], label="Diffusion", color="blue")
    plt.plot(traj_cfm[:, 0], traj_cfm[:, 1], label="CFM", color="red")
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
    pos_error_diff = np.array(pos_error_m1)
    pos_error_cfm = np.array(pos_error_m2)
    plt.plot(pos_error_diff, label="Diffusion", color="blue")
    plt.plot(pos_error_cfm, label="CFM", color="red")
    plt.title("Positional Error Comparison")
    plt.xlabel("Time Step")
    plt.ylabel("Error (Euclidean Distance)")
    plt.legend()
    plt.grid()
    error_plot_path = join(run_savepath, 'pos_error_comp.png')
    plt.savefig(error_plot_path)
    plt.close()

trajectory_data_diff, pos_error_data_diff = compare_euclid_pos_error("Diffusion")
trajectory_data_cfm, pos_error_data_cfm = compare_euclid_pos_error("CFM")

trajectory_m1 = trajectory_data_diff["Diffusion"]
trajectory_m2 = trajectory_data_cfm["CFM"]
pos_error_m1 = pos_error_data_diff["Diffusion"]
pos_error_m2 = pos_error_data_cfm["CFM"]

plot_savepath = 'logs/maze2d_umaze-v1/plans/release_H128_T64_LimitsNormalizer_b1_condFalse/'

save_plots(plot_savepath, trajectory_m1, trajectory_m2, pos_error_m1, pos_error_m2)
print(f"Done performing pos error comp.")
