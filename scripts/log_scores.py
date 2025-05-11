import subprocess
import csv
import os
import statistics

def run_plan_maze():
    # Run the plan_maze2d.py script and capture the output
    result = subprocess.run(['python', 'scripts/plan_maze2d.py'], capture_output=True, text=True)
    return result

def log_scores(num_iterations):
    scores_and_rewards = []

    for i in range(num_iterations):
        print(f"Running iteration {i + 1}/{num_iterations}...")
        result = run_plan_maze()
        
        # Extract the score and reward from the output
        score, reward = None, None
        for line in result.stdout.splitlines():
            if "score:" in line:
                score = float(line.split("score: ")[1].strip())
            if "R:" in line:  # Extract total reward
                reward = float(line.split("R: ")[1].split("|")[0].strip())
            if score is not None and reward is not None:
                scores_and_rewards.append((i + 1, score, reward))
                break

    # Ensure the logs directory exists
    os.makedirs('logs', exist_ok=True)

    # Write scores and rewards to the CSV file
    with open('logs/scores.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if os.stat('logs/scores.csv').st_size == 0:  # Write header if the file is empty
            writer.writerow(['Iteration', 'Score', 'Reward'])
        writer.writerows(scores_and_rewards)

    # Calculate and display statistics
    calculate_statistics(scores_and_rewards)

def calculate_statistics(scores_and_rewards):
    scores = [entry[1] for entry in scores_and_rewards]
    rewards = [entry[2] for entry in scores_and_rewards]

    mean_score = statistics.mean(scores)
    median_score = statistics.median(scores)
    mean_reward = statistics.mean(rewards)
    median_reward = statistics.median(rewards)

    print(f"\nStatistics:")
    print(f"Mean Score: {mean_score:.4f}, Median Score: {median_score:.4f}")
    print(f"Mean Reward: {mean_reward:.4f}, Median Reward: {median_reward:.4f}")

if __name__ == "__main__":
    num_iterations = 100  # Set the number of iterations you want to run
    log_scores(num_iterations)