import os
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

def get_euclid_storage():
    pos_error_data = {
        "Diffusion": [],
        "CFM": []
    }
    return pos_error_data

def compare_euclid_pos_error(method_name):
    # Storage:
    pos_error_data = get_euclid_storage()

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
    dataset_type = args.dataset.split("-")[1]

    diffusion_exp = utils.load_diffusion(
        args.logbase, 
        args.dataset, 
        args.diffusion_loadpath, 
        epoch=760000
        )
    
    diffusion = diffusion_exp.ema
    dataset = diffusion_exp.dataset
    
    policy = Policy(diffusion, dataset.normalizer)
    observation = env.reset()

    if args.conditional:
        print('Resetting target')
        env.set_target()

    pos_err_path = join(args.savepath, f"pos_err")
    os.makedirs(pos_err_path, exist_ok=True)
    
    if args.dataset == "maze2d-umaze-v1":
        env.set_state(np.array([3.90455573, 2.04127807]), np.array([0.15673357, -0.01839132]))
    # elif args.dataset == "maze2d-medium-v1":
    #     env.set_state(np.array([2.926775177, 1.956217731]), np.array([-0.02120387, -0.09685921]))
    # elif args.dataset == "maze2d-large-v1":
    #     env.set_state(np.array([0.94333326, 1.09938711]), np.array([0.10727024, 0.05407418]))

    target = env._target
    cond = {
        0 : observation, # Shape (4,)
        diffusion.horizon - 1: np.array([*target, 0, 0]), # [1 1 0 0], [6, 6, 0, 0], [7, 9, 0, 0]
    }
    action, samples = policy(cond, batch_size=args.batch_size)
    sequence = samples.observations[0] # (Horizon: 128, observation_dim: 4), len = 128     

    pos_error = []

    for t in range(env.max_episode_steps):
        state = env.state_vector().copy()

        if t < len(sequence) - 1:
            next_waypoint = sequence[t + 1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

        next_observation, reward, terminal, _ = env.step(action)

        pos_error.append(np.linalg.norm(next_observation[:2] - target[:2]))
        pos_error_data[method_name] = pos_error

    return pos_error_data, dataset_type

def save_plots(savepath, pos_error_m1, pos_error_m2, dataset):
    assert pos_error_m1 is not None, "pos_error_m1 is None"
    assert pos_error_m2 is not None, "pos_error_m2 is None"

    os.makedirs(savepath, exist_ok=True)

    # Plot Positional Error for both methods
    plt.figure(figsize=(8, 6))
    pos_error_diff = np.array(pos_error_m1)
    pos_error_cfm = np.array(pos_error_m2)
    plt.plot(pos_error_diff, label="Diffusion", color="blue")
    plt.plot(pos_error_cfm, label="CFM", color="red")
    plt.title(f"Positional Error Comparison {dataset}")
    plt.xlabel("Episode Step")
    plt.ylabel("Error (Euclidean Distance)")
    plt.legend()
    plt.grid()
    error_plot_path = join(savepath, f'pos_error_comp_{dataset}.png')
    plt.savefig(error_plot_path)
    plt.close()
    print(f"Error comp plot saved to: {error_plot_path}")

pos_error_data_diff, dataset = compare_euclid_pos_error("Diffusion")
pos_error_data_cfm, dataset = compare_euclid_pos_error("CFM")

pos_error_m1 = pos_error_data_diff["Diffusion"]
pos_error_m2 = pos_error_data_cfm["CFM"]

# Log or print the final positional error for both models
final_pos_error_diff = pos_error_m1[-1] if pos_error_m1 else None
final_pos_error_cfm = pos_error_m2[-1] if pos_error_m2 else None

print(f"Final positional error for Diffusion: {final_pos_error_diff}")
print(f"Final positional error for CFM: {final_pos_error_cfm}")

plot_savepath = 'logs/pos_error_comps/'
save_plots(plot_savepath, pos_error_m1, pos_error_m2, dataset)