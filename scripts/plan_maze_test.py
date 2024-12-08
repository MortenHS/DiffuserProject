import os
import csv
import json
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# # Logger and Environment Setup
env = datasets.load_environment(args.dataset)

# Load Diffusion Model
diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)
print(f"Loading diffusion from: {join(args.logbase, args.dataset, args.diffusion_loadpath)}")

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

# Overwrite the CSV file and set up headers
csv_path = join(args.savepath, 'results.csv')
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['run', 'score', 'total_reward', 'steps', 'terminal', 'epoch_diffusion'])

# Main Loop for Multiple Runs
num_runs = 100  # Increase the number of runs to 100
for run in range(num_runs):
    run_savepath = join(args.savepath, f'run_{run}')
    os.makedirs(run_savepath, exist_ok=True)

    observation = env.reset()

    if args.conditional:
        env.set_target()

    target = env._target
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    rollout = [observation.copy()]
    total_reward = 0
    observations = []

    for t in range(env.max_episode_steps):
        state = env.state_vector().copy()

        if t == 0:
            cond[0] = observation
            action, samples = policy(cond, batch_size=args.batch_size)
            actions = samples.actions[0]
            sequence = samples.observations[0]

            # Save the initial image (runX_0.png)
            initial_img_path = join(run_savepath, f'run{run}_0.png')
            renderer.composite(initial_img_path, samples.observations, ncol=1)

        if t < len(sequence) - 1:
            next_waypoint = sequence[t + 1]
        # --------------------------------------------------------------------------------------
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0

        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        # --------------------------------------------------------------------------------------

        # Use actions defined in the process, instead of using next_waypoint
        # --------------------------------------------------------------------------------------
        # else:
        #     actions = actions[1:]
        #     if len(actions) > 1:
        #         action = actions[0]
        #     else:
        #         # action = np.zeros(2)
        #         action = -state[2:]
        #         # pdb.set_trace() # Leads to (pdb) python debugger interrupt each step.
        # --------------------------------------------------------------------------------------

        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        score = env.get_normalized_score(total_reward)

        if terminal:
            break

        observation = next_observation
        observations.append(observation)
        rollout.append(next_observation.copy())

    # Save the final rollout image (runX_rollout.png)
    rollout_img_path = join(run_savepath, f'run{run}_rollout.png')
    renderer.composite(rollout_img_path, np.array(rollout)[None], ncol=1)

    # Save Rollout Data as JSON
    json_path = join(run_savepath, 'rollout.json')
    json_data = {
        'score': score,
        'step': t,
        'return':total_reward,
        'term': terminal,
        'epoch_diffusion': diffusion_experiment.epoch,
    }
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)

    # Append Results to CSV
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            score,        
            total_reward,  
            t,                      # Number of steps (integer)
            terminal,               # Terminal state (boolean)
            diffusion_experiment.epoch
        ])