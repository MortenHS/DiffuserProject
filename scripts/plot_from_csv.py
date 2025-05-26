import matplotlib.pyplot as plt
import csv
import os
import re
import pandas as pd
import numpy as np

def plot_loss(csv_file="loss_log.csv", savepath="logs/tests/loss_plot.png"):
    """
    Reads loss values from a CSV file and plots loss vs. training steps.

    Parameters:
        csv_file (str): Path to the CSV file containing loss data.
    """
    steps = []
    losses = []

    # Read the CSV file
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Skip header rows
            if row[0] == "Step" and row[1] == "Loss":
                continue
            try:
                steps.append(int(row[0]))  # Step
                losses.append(float(row[1]))  # Loss
            except ValueError:
                # Skip rows that cannot be parsed
                continue

    # Plot the loss vs. steps
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label="Training Loss", color="blue")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Loss vs. Training Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig(savepath)
    plt.close()

def plot_positional_errors(csv_path, save_path):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Drop rows where Model is NaN or doesn't contain '_'
    df = df[df['Model'].notna()]
    df = df[df['Model'].str.contains('_')]

    # Extract Method and Group
    df['Method'] = df['Model'].apply(lambda x: x.split('_')[0])
    df['Group'] = df['Model'].apply(lambda x: x.split('_')[1])

    # Define desired group order
    group_order = ['umaze', 'medium', 'large']
    method_order = ['Diffusion', 'CFM']

    # Sort by group and method
    df['Group'] = pd.Categorical(df['Group'], categories=group_order, ordered=True)
    df['Method'] = pd.Categorical(df['Method'], categories=method_order, ordered=True)
    df = df.sort_values(by=['Group', 'Method'])

    # Set up x-axis positions
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, df["Average Positional Error"], width, label='Average Error')
    ax.bar(x + width/2, df["Median Positional Error"], width, label='Median Error')

    ax.set_ylabel('Error')
    ax.set_xlabel('Model')
    ax.set_title('Average and Median Positional Errors, Iterations: 200')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_epoch_progression(csv_path, savepath):
    df = pd.read_csv(csv_path)

    # Ensure 'Epoch' is treated as integer for sorting
    df['Epoch'] = df['Epoch'].astype(int)

    # Sort by epoch (ascending)
    df = df.sort_values('Epoch')

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['Epoch'], df['Mean Score'], label='Mean Score', marker='o')
    plt.plot(df['Epoch'], df['Median Score'], label='Median Score', marker='o')
    plt.plot(df['Epoch'], df['Mean Reward'], label='Mean Reward', marker='o')
    plt.plot(df['Epoch'], df['Median Reward'], label='Median Reward', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Progression of Scores and Rewards Across Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

def compute_score_reward_stats(log_path):
    scores = []
    rewards = []
    pattern = re.compile(r"Extracted score: ([\-\d\.]+), reward: ([\-\d\.]+)")

    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                score = float(match.group(1))
                reward = float(match.group(2))
                scores.append(score)
                rewards.append(reward)

    scores = np.array(scores)
    rewards = np.array(rewards)

    print(f"Scores: mean={scores.mean():.2f}, median={np.median(scores):.2f}")
    print(f"Rewards: mean={rewards.mean():.2f}, median={np.median(rewards):.2f}")

    return {
        "score_mean": scores.mean(),
        "score_median": np.median(scores),
        "reward_mean": rewards.mean(),
        "reward_median": np.median(rewards),
    }

def plot_all_scores_vs_n(csv_path, save_dir, plot_rewards=False):
    df = pd.read_csv(csv_path)
    df['Dataset'] = df['Dataset'].str.strip()
    df['Model'] = df['Model'].str.strip()
    df['N'] = df['N'].astype(int)
    datasets = df['Dataset'].unique()
    models = df['Model'].unique()

    # Assign colors by model, not dataset
    model_colors = {'cfm': 'tab:blue', 'diffusion': 'tab:orange'}
    linestyles = {'cfm': '-', 'diffusion': '--'}
    markers = {'cfm': 'o', 'diffusion': 's'}

    os.makedirs(save_dir, exist_ok=True)

    for dataset in datasets:
        plt.figure(figsize=(10, 6))
        for model in models:
            sub = df[(df['Model'] == model) & (df['Dataset'] == dataset)]
            if sub.empty:
                continue
            x = sub['N']
            y = sub['Mean Score']
            color = model_colors.get(model, None)
            # Scatter plot
            plt.scatter(
                x, y,
                marker=markers.get(model, 'o'),
                color=color,
                alpha=0.7,
                label=f"{model} (Mean Score)"
            )
            # Regression line
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(
                    x, p(x),
                    color=color,
                    linestyle=linestyles.get(model, '-'),
                    linewidth=2,
                    label=f"{model} regression"
                )
            # Optionally plot rewards
            if plot_rewards:
                y_reward = sub['Mean Reward']
                plt.scatter(
                    x, y_reward,
                    marker=markers.get(model, 'x'),
                    color=color,
                    alpha=0.5,
                    label=f"{model} (Mean Reward)"
                )
                if len(x) > 1:
                    z_reward = np.polyfit(x, y_reward, 1)
                    p_reward = np.poly1d(z_reward)
                    plt.plot(
                        x, p_reward(x),
                        color=color,
                        linestyle=':',
                        linewidth=2,
                        label=f"{model} reward regression"
                    )

        plt.xlabel('Sampling steps [N]')
        plt.ylabel('Score' + (' / Reward' if plot_rewards else ''))
        plt.title(f'Scores vs. Sampling Steps ({dataset})')
        plt.legend(loc='best', fontsize='small')
        plt.grid(True)
        plt.tight_layout()
        savepath = os.path.join(save_dir, f'scores_vs_n_{dataset}.png')
        plt.savefig(savepath)
        plt.close()
        print(f"Plot saved to {savepath}")


if __name__ == "__main__":
    # For loss values of model training:
    # savepath="logs/tests/loss_plot.png"
    # plot_loss("slurms/loss_log.csv", savepath=savepath)
    # print(f"Loss plot generated successfully to {savepath}")

    # For positional errors:
    # csv_path = "/cluster/work/mortenhs/Janner/diffuser/logs/pos_error_results.csv"
    # savepath_positional = "/cluster/work/mortenhs/Janner/diffuser/logs/pos_error_comps/positional_errors_plot.png"
    # plot_positional_errors(csv_path, savepath_positional)
    # print(f"Positional error plot generated successfully to {savepath_positional}")

    # For epoch progression:
    # plot_epoch_progression('logs/scores.csv', 'logs/tests/diff_scores_epoch_test.png')

    # For scores from log file:
    # stats = compute_score_reward_stats("logs/log_scores.log")
    # print(stats)

    # plot_scores_vs_n('logs/scores_cfm.csv', 'logs/plots_from_tests/cfm_scores_vs_n.png')
    # plot_scores_vs_n('logs/scores_diff.csv', 'logs/plots_from_tests/diff_scores_vs_n.png')
    # plot_scores_vs_n('logs/sampling_step_scores.csv', 'logs/plots_from_tests/samp_steps.png')

    plot_all_scores_vs_n('logs/sampling_step_scores.csv', 'logs/plots_from_tests')
