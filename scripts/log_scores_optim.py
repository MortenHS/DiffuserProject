import os
import csv
import time
import statistics
from concurrent.futures import ProcessPoolExecutor
from plan_maze2d_optim import Parser, plan_maze


def run_plan_maze_parallel(args, num_iterations):
    """
    Runs the maze planning logic in parallel for multiple iterations.
    """
    scores = []
    rewards = []

    for _ in range(num_iterations):
        score, reward = plan_maze(args)
        scores.append(score)
        rewards.append(reward)

    mean_score = statistics.mean(scores)
    median_score = statistics.median(scores)
    mean_reward = statistics.mean(rewards)
    median_reward = statistics.median(rewards)

    model = "cfm" if args.config.endswith("_cfm") else "diffusion"
    dataset_name = args.dataset.split("-")[1]  # Extract umaze, medium, or large

    return [model, dataset_name, f"{mean_score:.2f}", f"{median_score:.2f}", f"{mean_reward:.2f}", f"{median_reward:.2f}"]


def log_scores(configs_and_datasets, num_iterations):
    """
    Logs scores and rewards for multiple configurations and datasets.
    """
    args = Parser().parse_args("plan")
    aggregated_results = []

    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor() as executor:
        futures = []
        for config, dataset in configs_and_datasets:
            # Update args for the current config and dataset
            args.config = config
            args.dataset = dataset

            # Submit the task to the executor
            futures.append(executor.submit(run_plan_maze_parallel, args, num_iterations))

        for future in futures:
            aggregated_results.append(future.result())

    # Ensure the logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Write aggregated results to the CSV file
    with open("logs/scores.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Dataset", "Mean Score", "Median Score", "Mean Reward", "Median Reward"])
        writer.writerows(aggregated_results)

    print("\nAggregated results have been written to 'logs/scores.csv'.")

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
    configs_and_datasets = [
        ("config.maze2d", "maze2d-umaze-v1"),
        # ("config.maze2d", "maze2d-medium-v1"),
        # ("config.maze2d", "maze2d-large-v1"),
        ("config.maze2d_cfm", "maze2d-umaze-v1"),
        # ("config.maze2d_cfm", "maze2d-medium-v1"),
        # ("config.maze2d_cfm", "maze2d-large-v1"),
    ]

    num_iterations = 2  # Set the number of iterations for each config and dataset
    start_time = time.time()
    log_scores(configs_and_datasets, num_iterations)
    generate_latex_table()
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds.")