import subprocess
import csv
import os
import time
import logging
import torch

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level to DEBUG for detailed output
    format="%(asctime)s - %(levelname)s - %(message)s",  
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/log_scores.log", mode="w")  
    ]
)

def run_plan_maze(config, dataset):
    """
    Run the plan_maze2d.py script with the specified config and dataset.
    """
    command = ['python', '/cluster/work/mortenhs/Janner/diffuser/scripts/plan_maze2d.py', '--config', config, '--dataset', dataset]
    result = subprocess.run(command, capture_output=True, text=True)
    # logging.debug(f"Subprocess stdout: {result.stdout}")
    # logging.debug(f"Subprocess stderr: {result.stderr}")
    return result

def log_scores(configs_and_datasets, num_iterations):
    """
    Run plan_maze2d.py for multiple configurations and datasets, compute average and median scores and rewards,
    and log the aggregated results.
    """
    # Define the max_episode_steps for each dataset type
    max_episode_steps = {
        'umaze': 299,
        'medium': 599,
        'large': 799
    }

    aggregated_results = []
    used_epoch = None

    for config, dataset in configs_and_datasets:
        logging.info(f"Starting processing for config: {config}, dataset: {dataset}")
        scores = torch.tensor([], device='cuda')
        rewards = torch.tensor([], device='cuda')

        # Extract the dataset type (e.g., umaze, medium, large)
        dataset_type = dataset.split('-')[1]
        t_value = max_episode_steps.get(dataset_type, None)

        if t_value is None:
            logging.error(f"Unknown dataset type: {dataset_type}")
            continue

        for i in range(num_iterations):
            logging.info(f"Running {config} on {dataset}, iteration {i + 1}/{num_iterations}")
            result = run_plan_maze(config, dataset)

            # Parse epoch from the output if not already set
            if used_epoch is None:
                for line in result.stdout.splitlines():
                    if "[ utils/serialization ] Loading model epoch:" in line:
                        try:
                            used_epoch = int(line.split("epoch:")[1].split("\\n")[0].strip())
                        except Exception:
                            used_epoch = "unknown"

            # Process only the last line with the correct t: value
            for line in result.stdout.splitlines():
                if line.startswith(f"t: {t_value}"):
                    reward = float(line.split("R: ")[1].split("|")[0].strip())
                    score = float(line.split("score: ")[1].split("|")[0].strip()) * 100  # Scale score by 100
                    scores = torch.cat((scores, torch.tensor([score], device='cuda')))
                    rewards = torch.cat((rewards, torch.tensor([reward], device='cuda')))
                    logging.debug(f"Extracted score: {score:.4f}, reward: {reward:.4f}")
                    break 

        # Compute average and median for the current config and dataset
        mean_score = torch.mean(scores).item() if scores.numel() > 0 else 0.0
        median_score = torch.median(scores).item() if scores.numel() > 0 else 0.0
        mean_reward = torch.mean(rewards).item() if rewards.numel() > 0 else 0.0
        median_reward = torch.median(rewards).item() if rewards.numel() > 0 else 0.0

        model = "cfm" if config == "config.maze2d_cfm" else "diffusion"
        logging.info(f"Finished processing {config} on {dataset}: "
                     f"Mean Score={mean_score:.2f}, Median Score={median_score:.2f}, "
                     f"Mean Reward={mean_reward:.2f}, Median Reward={median_reward:.2f}")
        
        aggregated_results.append([model, used_epoch, dataset_type, f"{mean_score:.2f}", f"{median_score:.2f}", f"{mean_reward:.2f}", f"{median_reward:.2f}"])

    os.makedirs('logs', exist_ok=True)

    # Write aggregated results to the CSV file
    csv_path = 'logs/scores.csv'
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Model','Epoch','Dataset','Mean Score', 'Median Score', 'Mean Reward', 'Median Reward'])
        writer.writerows(aggregated_results)

def generate_latex_table(csv_file='logs/scores.csv', output_file='logs/latex_table.txt', num_iterations=1):
    """
    Reads the aggregated scores and rewards from the CSV file and generates a LaTeX table.
    Stores the LaTeX table in a text file.
    """
    if not os.path.exists(csv_file):
        logging.error(f"CSV file '{csv_file}' not found.")
        return

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        data = list(reader)

    header = data[0]
    rows = data[1:]

    latex_lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\begin{tabular}{@{}l l r r r r@{}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    latex_lines.extend(" & ".join(row) + " \\\\" for row in rows)
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Aggregated Scores and Rewards, N=[64, 256, 256] (Averaged over {num_iterations} iterations)}}",
        "\\label{tab:scores_rewards}",
        "\\end{table}",
    ])

    os.makedirs('logs', exist_ok=True)
    with open(output_file, mode='w') as file:
        file.write("\n".join(latex_lines))

    logging.info(f"LaTeX table has been saved to '{output_file}'.")

if __name__ == "__main__":
    configs_and_datasets = [
        # ('config.maze2d', 'maze2d-umaze-v1'),
        # ('config.maze2d', 'maze2d-medium-v1'),
        # ('config.maze2d', 'maze2d-large-v1'),
        ('config.maze2d_cfm', 'maze2d-umaze-v1'),
        # ('config.maze2d_cfm', 'maze2d-medium-v1'),
        # ('config.maze2d_cfm', 'maze2d-large-v1'),
    ]

    num_iterations = 20

    start_time = time.time()
    log_scores(configs_and_datasets, num_iterations)
    # generate_latex_table(num_iterations=num_iterations)
    end_time = time.time()
    logging.info(f"Total time taken: {end_time - start_time:.2f} seconds")