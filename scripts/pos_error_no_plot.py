import os
import csv
import numpy as np
from os.path import join

from diffuser.guides.policies import Policy
import time
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

def calculate_pos_error(method_name, iterations):
    pos_error_data = []
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
    diffusion = diffusion_exp.ema
    dataset = diffusion_exp.dataset

    policy = Policy(diffusion, dataset.normalizer)
    for _ in range(iterations):
        observation = env.reset()
        target = env._target
        cond = {
            diffusion.horizon - 1: np.array([*target, 0, 0])
        }
        pos_error = []
        cond[0] = observation
        action, samples = policy(cond, batch_size=args.batch_size)
        actions = samples.actions[0]
        sequence = samples.observations[0]
        for t in range(env.max_episode_steps):
            state = env.state_vector().copy()

            if t < len(sequence) - 1:
                next_waypoint = sequence[t + 1]
            else:
                next_waypoint = sequence[-1].copy()
                next_waypoint[2:] = 0
            action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

            next_observation, _, terminal, _ = env.step(action)
            pos_error.append(np.linalg.norm(next_observation[:2] - target[:2]))

            if terminal:
                break

            observation = next_observation

        pos_error_data.append(pos_error[-1] if pos_error else None)
    dataset_name = args.dataset.split("-")[1]
    return pos_error_data, dataset_name

def log_results_to_csv(filepath, results):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Average Positional Error", "Median Positional Error"])
        for model, avg_error, median_error in results:
            writer.writerow([model, avg_error, median_error])

def main(iterations=100, output_file="logs/pos_error_results.csv"):
    # Calculate positional errors for both models
    pos_error_diff, dataset_name = calculate_pos_error("Diffusion", iterations)
    pos_error_cfm, dataset_name = calculate_pos_error("CFM", iterations)

    # Compute average and median positional errors
    avg_pos_error_diff = np.mean([e for e in pos_error_diff if e is not None])
    median_pos_error_diff = np.median([e for e in pos_error_diff if e is not None])

    avg_pos_error_cfm = np.mean([e for e in pos_error_cfm if e is not None])
    median_pos_error_cfm = np.median([e for e in pos_error_cfm if e is not None])

    # Prepare results for logging
    results = [
        (f"Diffusion_{dataset_name}", avg_pos_error_diff, median_pos_error_diff),
        (f"CFM_{dataset_name}", avg_pos_error_cfm, median_pos_error_cfm)
    ]

    # Log results to CSV
    start_time = time.time()

    log_results_to_csv(output_file, results)

    elapsed_time = time.time() - start_time
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow(["Time Elapsed (seconds)", elapsed_time])
    print(f"Results logged to {output_file}")

if __name__ == "__main__":
    main(iterations=1000)