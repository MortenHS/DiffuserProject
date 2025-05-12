import subprocess
import csv
import os
import statistics
import time
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level to DEBUG for detailed output
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.StreamHandler(),  # Log to the terminal
        logging.FileHandler("logs/log_scores.log", mode="w")  # Log to a file
    ]
)

def run_plan_maze(config, dataset):
    """
    Run the plan_maze2d.py script with the specified config and dataset.
    """
    command = ['python', 'scripts/plan_maze2d_optim.py', '--config', config, '--dataset', dataset]
    result = subprocess.run(command, capture_output=True, text=True)
    return result

def log_scores(configs_and_datasets, num_iterations):
    """
    Run plan_maze2d.py for multiple configurations and datasets, compute average and median scores and rewards,
    and log the aggregated results.
    """
    aggregated_results = []

    for config, dataset in configs_and_datasets:
        logging.info(f"Starting processing for config: {config}, dataset: {dataset}")
        scores = []
        rewards = []

        for i in range(num_iterations):
            logging.info(f"Running {config} on {dataset}, iteration {i + 1}/{num_iterations}")
            result = run_plan_maze(config, dataset)
            logging.debug(f"Processing output for {config}, {dataset}, iteration {i + 1}")
            # Extract the score and reward from the output
            for line in result.stdout.splitlines():
                logging.debug(f"Parsing line: {line}")
                if "t:" in line and "score:" in line and "R:" in line:
                    reward = float(line.split("R: ")[1].split("|")[0].strip())
                    score = float(line.split("score: ")[1].split("|")[0].strip()) * 100  # Scale score by 100
                    scores.append(score)
                    rewards.append(reward)
                    logging.debug(f"Extracted score: {score}, reward: {reward}")
                    break

        # Compute average and median for the current config and dataset
        mean_score = statistics.mean(scores)
        median_score = statistics.median(scores)
        mean_reward = statistics.mean(rewards)
        median_reward = statistics.median(rewards)

        model = "cfm" if config == "config.maze2d_cfm" else "diffusion"
        dataset_name = dataset.split('-')[1]  # Extract umaze, medium, or large
        logging.info(f"Finished processing {config} on {dataset}: "
                f"Mean Score={mean_score:.2f}, Median Score={median_score:.2f}, "
                f"Mean Reward={mean_reward:.2f}, Median Reward={median_reward:.2f}")
        # Store the aggregated results
        aggregated_results.append([model, dataset_name, f"{mean_score:.2f}", f"{median_score:.2f}", f"{mean_reward:.2f}", f"{median_reward:.2f}"])

    # Ensure the logs directory exists
    os.makedirs('logs', exist_ok=True)

    # Write aggregated results to the CSV file
    with open('logs/scores.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Dataset', 'Mean Score', 'Median Score', 'Mean Reward', 'Median Reward'])
        writer.writerows(aggregated_results)

def generate_latex_table(csv_file='logs/scores.csv', output_file='logs/latex_table.txt'):
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
        "\\caption{Aggregated Scores and Rewards for Maze2D Experiments}",
        "\\label{tab:scores_rewards}",
        "\\end{table}",
    ])

    os.makedirs('logs', exist_ok=True)
    with open(output_file, mode='w') as file:
        file.write("\n".join(latex_lines))

    logging.info(f"LaTeX table has been saved to '{output_file}'.")

if __name__ == "__main__":
    # Define the configurations and datasets to run
    configs_and_datasets = [
        ('config.maze2d', 'maze2d-umaze-v1'),
        ('config.maze2d', 'maze2d-medium-v1'),
        ('config.maze2d', 'maze2d-large-v1'),
        ('config.maze2d_cfm', 'maze2d-umaze-v1'),
        ('config.maze2d_cfm', 'maze2d-medium-v1'),
        ('config.maze2d_cfm', 'maze2d-large-v1'),
        # Add more (config, dataset) pairs as needed
    ]

    num_iterations = 2  # Set the number of iterations for each config and dataset

    start_time = time.time()
    logging.info("Starting the score logging process.")
    log_scores(configs_and_datasets, num_iterations)
    generate_latex_table()
    end_time = time.time()
    logging.info(f"Total time taken: {end_time - start_time:.2f} seconds")