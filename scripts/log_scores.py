import subprocess
import csv
import os
import statistics
import time

def run_plan_maze(config, dataset):
    """
    Run the plan_maze2d.py script with the specified config and dataset.
    """
    command = ['python', 'scripts/plan_maze2d.py', '--config', config, '--dataset', dataset]
    result = subprocess.run(command, capture_output=True, text=True)
    return result

def log_scores(configs_and_datasets, num_iterations):
    """
    Run plan_maze2d.py for multiple configurations and datasets, compute average and median scores and rewards,
    and log the aggregated results.
    """
    aggregated_results = []

    for config, dataset in configs_and_datasets:
        print(f"Running for config: {config}, dataset: {dataset}")
        scores = []
        rewards = []

        for i in range(num_iterations):
            print(f"  Iteration {i + 1}/{num_iterations}...")
            result = run_plan_maze(config, dataset)

            # Extract the score and reward from the output
            for line in result.stdout.splitlines():
                if "t:" in line and "score:" in line and "R:" in line:
                    reward = float(line.split("R: ")[1].split("|")[0].strip())
                    score = float(line.split("score: ")[1].split("|")[0].strip()) * 100  # Scale score by 100
                    scores.append(score)
                    rewards.append(reward)
                    break

        # Compute average and median for the current config and dataset
        mean_score = statistics.mean(scores)
        median_score = statistics.median(scores)
        mean_reward = statistics.mean(rewards)
        median_reward = statistics.median(rewards)

        model = "cfm" if config == "config.maze2d_cfm" else "diffusion"
        dataset_name = dataset.split('-')[1]  # Extract umaze, medium, or large

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
        print(f"CSV file '{csv_file}' not found.")
        return

    # Read data from the CSV file
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        data = list(reader)

    # Extract header and rows
    header = data[0]  # ['Model', 'Dataset', 'Mean Score', 'Median Score', 'Mean Reward', 'Median Reward']
    rows = data[1:]  # Skip the header row

    # Generate LaTeX table
    latex_table = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{@{}l l r r r r@{}}\n\\toprule\n"
    latex_table += " & ".join(header) + " \\\\\n\\midrule\n"

    for row in rows:
        latex_table += " & ".join(row) + " \\\\\n"

    latex_table += "\\bottomrule\n\\end{tabular}\n\\caption{Aggregated Scores and Rewards for Maze2D Experiments}\n\\label{tab:scores_rewards}\n\\end{table}"

    # Ensure the logs directory exists
    os.makedirs('logs', exist_ok=True)

    # Write the LaTeX table to a text file
    with open(output_file, mode='w') as file:
        file.write(latex_table)

    print(f"LaTeX table has been saved to '{output_file}'.")

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
    log_scores(configs_and_datasets, num_iterations)
    generate_latex_table()
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")