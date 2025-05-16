import matplotlib.pyplot as plt
import csv
import os
import re
import pandas as pd
import numpy as np

def plot_loss_from_csv(csv_file="loss_log.csv", savepath="logs/tests/loss_plot.png"):
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

def plot_positional_errors_from_csv(csv_path, save_path):
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


if __name__ == "__main__":
    # For loss values of model training:
    # savepath="logs/tests/loss_plot.png"
    # plot_loss_from_csv("logs/tests/loss_log.csv", savepath=savepath)
    # print(f"Loss plot generated successfully to {savepath}")

    # For positional errors:
    # csv_path = "/cluster/work/mortenhs/Janner/diffuser/logs/pos_error_results.csv"
    # savepath_positional = "/cluster/work/mortenhs/Janner/diffuser/logs/pos_error_comps/positional_errors_plot.png"
    # plot_positional_errors_from_csv(csv_path, savepath_positional)
    # print(f"Positional error plot generated successfully to {savepath_positional}")

    # For epoch progression:
    plot_epoch_progression('logs/umaze_scores_epoch_test.csv', 'logs/tests/umaze_scores_epoch_test.png')