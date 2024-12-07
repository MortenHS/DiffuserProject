import json
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)
print(f"Loading diffusion from: {join(args.logbase, args.dataset, args.diffusion_loadpath)}")

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

#---------------------------------- main loop ----------------------------------#

observation = env.reset()

if args.conditional:
    print('Resetting target')
    env.set_target()

## set conditioning xy position to be the goal
target = env._target
cond = {
    diffusion.horizon - 1: np.array([*target, 0, 0]),
}

## observations for rendering
rollout = [observation.copy()]
used_actions = []
observations = []
total_reward = 0
for t in range(env.max_episode_steps):

    state = env.state_vector().copy()

    ## can replan if desired, but the open-loop plans are good enough for maze2d
    ## that we really only need to plan once
    if t == 0:
        cond[0] = observation
        action, samples = policy(cond, batch_size=args.batch_size) # Process defined actions/policy actions
        actions = samples.actions[0]
        sequence = samples.observations[0] # 384 elements
    # pdb.set_trace()

    # ####
    if t < len(sequence) - 1:
        next_waypoint = sequence[t+1]
    
    ## If we want to use calculated actions:
    # --------------------------------------------------------------------------------------
    else:
        next_waypoint = sequence[-1].copy()
        next_waypoint[2:] = 0
        # pdb.set_trace()
        # if t == env.max_episode_steps - 3:
        #     print(f"\nNext_waypoint: {next_waypoint}\n")
        #     print(f"Sequence: {sequence}\n")
            
    # can use actions or define a simple controller based on state predictions
    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:]) # Calculated actions
    # --------------------------------------------------------------------------------------
    # pdb.set_trace()
    ####

    # Use actions defined in the process, instead of using next_waypoint
    # --------------------------------------------------------------------------------------
    # else:
    #     actions = actions[1:]
    #     if len(actions) > 1:
    #         action = actions[0]
    #     else:
    #         # action = np.zeros(2)
    #         action = -state[2:]
    #         # pdb.set_trace() # Leads to (pdb) python debugger interrupt each step.
    #--------------------------------------------------------------------------------------
    
    # Section 4.3.1 Scoring refers to this:
    next_observation, reward, terminal, _ = env.step(action)
    used_actions.append(action)
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'{action}'
    )

    if 'maze2d' in args.dataset:
        xy = next_observation[:2]
        goal = env.unwrapped._target
        print(
            f'maze | pos: {xy} | goal: {goal}'
        )

    ## update rollout observations
    rollout.append(next_observation.copy())

    # logger.log(score=score, step=t)
    if t % args.vis_freq == 0 or terminal:
        fullpath = join(args.savepath, f'{t}.png')

        if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)


        # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

        ## save rollout thus far
        # renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)
        renderer.composite(join(args.savepath, 'rollout_' + str(t) + '.png'), np.array(rollout)[None], ncol=1) # Makes the complete path(old rollout.png) now be the rollout_ + final t value + .png

        # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

    if terminal:
        break

    observation = next_observation
    observations.append(observation)
# Length of used actions = 800.

# logger.finish(t, env.max_episode_steps, score=score, value=0)

def plot_observation_points(observations_to_plot, target):
    observation_x = [point[0] for point in observations_to_plot]
    observation_y = [point[1] for point in observations_to_plot]
    plt.figure(figsize=(8, 6))
    plt.plot(observation_x, observation_y, label="Observations", color='red', linewidth=1)
    # Mark the first and last actions with points
    plt.scatter(observation_x[0], observation_y[0], color='blue', label='Starting point', zorder=5)
    plt.scatter(observation_x[-1], observation_y[-1], color='green', label='End point', zorder=5)
    plt.scatter(target[0], target[1], color='gold', label='Target point', zorder=5)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title('Observation points', fontsize=14)
    plt.xlabel('Observations_x', fontsize=12)
    plt.ylabel('Observations_y', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plotted_observation_points.jpeg", format='jpeg', dpi=300)

# def plot_observation_vels(observations_to_plot):
#     observation_vel_x = [point[2] for point in observations_to_plot]
#     observation_vel_y = [point[3] for point in observations_to_plot]
#     plt.figure(figsize=(8, 6))
#     plt.plot(observation_vel_x, observation_vel_y, label="Observations", color='red', linewidth=1)
#     # Mark the first and last actions with points
#     plt.scatter(observation_vel_x[0], observation_vel_y[0], color='blue', label='Starting velocity', zorder=5)
#     plt.scatter(observation_vel_x[-1], observation_vel_y[-1], color='green', label='End velocity', zorder=5)
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.title('Observation velocities', fontsize=14)
#     plt.xlabel('Observations_x', fontsize=12)
#     plt.ylabel('Observations_y', fontsize=12)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("plotted_observation_vels.jpeg", format='jpeg', dpi=300)


def plot_actions_vs_time(actions_to_plot, max_timesteps=400):
    actions_to_plot = actions_to_plot[:max_timesteps]
    steps = list(range(len(actions_to_plot)))
    actions_x = [point[0] for point in actions_to_plot]
    actions_y = [point[1] for point in actions_to_plot]

    plt.figure(figsize=(8, 6))
    # Plot the main line for all actions over time (steps)
    plt.plot(steps, actions_x, label="Actions in X", color='red', linewidth=0.5)
    plt.plot(steps, actions_y, label="Actions in Y", color='blue', linewidth=0.5)

    # Mark the first and last actions on the plot
    plt.scatter(steps[0], actions_x[0], color='orange', label='First Action (X)', zorder=5)
    plt.scatter(steps[0], actions_y[0], color='purple', label='First Action (Y)', zorder=5)
    plt.scatter(steps[-1], actions_x[-1], color='green', label='Last Action (X)', zorder=5)
    plt.scatter(steps[-1], actions_y[-1], color='cyan', label='Last Action (Y)', zorder=5)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title('Actions Over Time', fontsize=14)
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Actions', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plotted_actions_vs_time.jpeg", format='jpeg', dpi=300)
    plt.show()

def moving_average(data, window_size):
    """Applies a moving average to smooth the data."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_actions_with_vectors(actions_to_plot, max_timesteps=400,window_size=10):
    actions_to_plot = actions_to_plot[:max_timesteps]
    actions_x = [point[0] for point in actions_to_plot]
    actions_y = [point[1] for point in actions_to_plot]
    smoothed_x = moving_average(actions_x, window_size)
    smoothed_y = moving_average(actions_y, window_size)

    # Calculate displacements for vectors
    dx = np.diff(smoothed_x)
    dy = np.diff(smoothed_y)

    # Start points of vectors for quiver (exclude the last point due to np.diff)
    start_x = smoothed_x[:-1]
    start_y = smoothed_y[:-1]
    steps = np.arange(len(dx))  # Steps for vector start points

    plt.figure(figsize=(8, 6))

    plt.quiver(
        start_x, start_y,    # Starting points of vectors
        dx, dy,              # Vector components
        scale_units='xy', angles='xy', scale=1, color='blue', alpha=0.7, width=0.002
    )

    # Highlight the start and end points
    plt.scatter(start_x[0], start_y[0], color='orange', label='Start Point', zorder=5)
    plt.scatter(start_x[-1], start_y[-1], color='green', label='End Point', zorder=5)

    # Add grid, title, and labels
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.title('Smoothed Trajectory with Vectors', fontsize=14)
    plt.xlabel('Actions in X', fontsize=12)
    plt.ylabel('Actions in Y', fontsize=12)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig("smoothed_vectorized_actions.jpeg", format='jpeg', dpi=300)
    plt.show()

plot_observation_points(actions,target)
# plot_observation_vels(observations)
# plot_actions_vs_time(used_actions)
# plot_actions_with_vectors(used_actions)

# save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
