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
    '''
    Plots the progression of scores and rewards across training steps from a CSV file.
    Training steps called 'Epoch' in the CSV.
    '''
    df = pd.read_csv(csv_path)

    df['Model'] = df['Model'].str.strip()
    # Determine model name
    model_name = df['Model'].iloc[0].lower()
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
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title(f'Progression of Scores and Rewards Across training steps for {model_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()
    print(f"Training step progression plot saved to {savepath}")

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

def plot_scores_for_tables(csv_path, save_dir):
    df = pd.read_csv(csv_path)
    df['Dataset'] = df['Dataset'].str.strip()
    df['Model'] = df['Model'].str.strip()
    # Determine model name
    model_name = df['Model'].iloc[0].lower()
    if 'diffusion' in model_name:
        fname = 'diffusion_scores_datasets.png'
    elif 'cfm' in model_name:
        fname = 'cfm_scores_datasets.png'
    else:
        fname = f'{model_name}_scores_datasets.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    # Create a grouped bar chart for each dataset, showing Mean Score and Mean Reward per experiment
    bar_width = 0.20
    experiments = df['Dataset']
    x = np.arange(len(df))

    plt.bar(x - bar_width/2, df['Mean Score'], width=bar_width, label='Mean Score', color='tab:blue')
    plt.bar(x + bar_width/2, df['Mean Reward'], width=bar_width, label='Mean Reward', color='tab:orange')
    # Write value on top of each bar
    for i, (score, reward) in enumerate(zip(df['Mean Score'], df['Mean Reward'])):
        plt.text(x[i] - bar_width/2, score, f"{score:.2f}", ha='center', va='bottom', fontsize=8)
        plt.text(x[i] + bar_width/2, reward, f"{reward:.2f}", ha='center', va='bottom', fontsize=8)
    plt.xticks(x, experiments, rotation=45, ha='right')
    plt.xlabel('Dataset')
    plt.ylabel('Value')
    plt.title(f'Mean scores and rewards for all datasets with {model_name}')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    savepath = os.path.join(save_dir, fname)
    plt.savefig(savepath)
    plt.close()
    print(f"Plot saved to {savepath}")
    
def plot_scores_vs_eachother(csv_path, save_dir): 
    df = pd.read_csv(csv_path)
    df = df.sort_values("Dataset")

    width = 0.35  # wider bars for clarity
    datasets = df["Dataset"].unique()
    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(10, 6))

    os.makedirs(save_dir, exist_ok=True)

    # Get values for each model
    values_diff = []
    values_cfm = []
    for dataset in datasets:
        row_diff = df[(df["Model"].str.lower() == "diffusion") & (df["Dataset"] == dataset)]
        row_cfm = df[(df["Model"].str.lower() == "cfm") & (df["Dataset"] == dataset)]
        values_diff.append(row_diff["Mean Score"].iloc[0] if not row_diff.empty else 0)
        values_cfm.append(row_cfm["Mean Score"].iloc[0] if not row_cfm.empty else 0)

    # Plot bars side by side
    bars1 = ax.bar(x - width/2, values_diff, width, label="Diffusion Mean Score", color="tab:blue")
    bars2 = ax.bar(x + width/2, values_cfm, width, label="CFM Mean Score", color="tab:orange")

    # Add value labels above each bar
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Score")
    ax.set_title("Model Scores by Dataset")
    ax.legend()
    plt.tight_layout()
    savepath = os.path.join(save_dir, f'h2h_scores.png')
    plt.savefig(savepath)
    plt.close()
    print(f"Plot saved to {savepath}")

def plot_select_sampling_scores(csv_path, save_dir, model, dataset):
    df = pd.read_csv(csv_path)
    df['Dataset'] = df['Dataset'].str.strip()
    df['Model'] = df['Model'].str.strip()
    df['N'] = df['N'].astype(int)

    # Filter for the selected model and dataset
    sub = df[(df['Model'].str.lower() == model.lower()) & (df['Dataset'].str.lower() == dataset.lower())]
    if sub.empty:
        print(f"No data found for model '{model}' and dataset '{dataset}'.")
        return

    sub = sub.sort_values('N')
    x = sub['N']
    y_score = sub['Mean Score']
    y_reward = sub['Mean Reward']

    plt.figure(figsize=(8, 5))
    plt.plot(x, y_score, marker='o', label='Mean Score', color='tab:blue')
    plt.plot(x, y_reward, marker='s', label='Mean Reward', color='tab:orange')
    plt.xlabel('Sampling steps [N]')
    plt.ylabel('Value')
    plt.title(f'Scores vs. Sampling Steps\nModel: {model}, Dataset: {dataset}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    fname = f"{model.lower()}_{dataset.lower()}_scores_vs_n.png"
    savepath = os.path.join(save_dir, fname)
    plt.savefig(savepath)
    plt.close()
    print(f"Plot saved to {savepath}")

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
            # Regression line (skip the first datapoint)
            if len(x) > 2:
                x_reg = x.iloc[1:]
                y_reg = y.iloc[1:]
                z = np.polyfit(x_reg, y_reg, 1)
                p = np.poly1d(z)
                plt.plot(
                    x_reg, p(x_reg),
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
                if len(x) > 2:
                    x_reg_r = x.iloc[1:]
                    y_reg_r = y_reward.iloc[1:]
                    z_reward = np.polyfit(x_reg_r, y_reg_r, 1)
                    p_reward = np.poly1d(z_reward)
                    plt.plot(
                        x_reg_r, p_reward(x_reg_r),
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
    '''
    Functions to plot various metrics from CSV files.
    Usage:
        - plot_loss: Plots loss values from a CSV file.
        - plot_positional_errors: Plots positional errors from a CSV file.
        - plot_epoch_progression: Plots scores and rewards progression across training steps.
        - compute_score_reward_stats: Computes and prints average scores and rewards from a log file.
        - plot_scores_for_tables: Plots mean scores and rewards for datasets in a bar chart.
        - plot_scores_vs_eachother: Plots scores of the different models against each other.
        - plot_select_sampling_scores: Plots scores for a specific model and dataset against sampling steps.
        - plot_all_scores_vs_n: Plots all scores vs. sampling steps for both models and all datasets.
    '''

    # For loss values of model training:
    # savepath="logs/tests/loss_plot.png"
    # plot_loss("slurms/loss_log.csv", savepath=savepath)
    # print(f"Loss plot generated successfully to {savepath}")

    # For positional errors:
    # csv_path = "/cluster/work/mortenhs/Janner/diffuser/logs/pos_error_results.csv"
    # savepath_positional = "/cluster/work/mortenhs/Janner/diffuser/logs/pos_error_comps/positional_errors_plot.png"
    # plot_positional_errors(csv_path, savepath_positional)
    # print(f"Positional error plot generated successfully to {savepath_positional}")

    # Plot scores over epochs:
    # plot_epoch_progression('logs/epoch_evaluations/diff_scores_epoch.csv', 'logs/epoch_evaluations/diff_epoch_progression.png')
    # plot_epoch_progression('logs/epoch_evaluations/cfm_scores_epoch.csv', 'logs/epoch_evaluations/cfm_epoch_progression.png')

    # Print average scores and rewards from log file:
    # stats = compute_score_reward_stats("logs/logfiles/log_scores.log")
    # print(stats)

    # Plot all sampling step scores:
    plot_all_scores_vs_n('logs/score_files/sampling_step_scores.csv', 'logs/plots_from_tests')

    # Plot select sampling step scores:
    # model_and_dataset_pairs = [
    #     ('diffusion', 'umaze'),
    #     ('diffusion', 'medium'),
    #     ('diffusion', 'large'),
    #     ('cfm', 'umaze'),
    #     ('cfm', 'medium'),
    #     ('cfm', 'large')
    # ]
    # for model, dataset in model_and_dataset_pairs:
    #     plot_select_sampling_scores('logs/score_files/sampling_step_scores.csv', 'logs/plots_from_tests', model, dataset)
    
    # plot_scores_vs_eachother('logs/score_files/table_scores_h2h.csv', 'logs/plots_from_tests')
    
    # Plot bar charts for table score values:
    # plot_scores_for_tables('logs/score_files/tablescores_diff.csv', 'logs/plots_from_tests')
    # plot_scores_for_tables('logs/score_files/tablescores_cfm.csv', 'logs/plots_from_tests')
    