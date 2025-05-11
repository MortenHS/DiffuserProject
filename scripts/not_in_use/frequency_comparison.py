import json
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import pdb
import os

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import visualize_d4rl
import mujoco_py
os.environ["MUJOCO_GL"] = "egl"

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)

if __name__ == "__main__":
    # Specify the environment name
    env_name = 'maze2d-large-v1'

    # Call the visualization function
    visualize_d4rl.visualize_env(env_name)

# #---------------------------------- loading ----------------------------------#

# diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)
# print(f"Loading diffusion from: {join(args.logbase, args.dataset, args.diffusion_loadpath)}")

# diffusion = diffusion_experiment.ema
# dataset = diffusion_experiment.dataset
# renderer = diffusion_experiment.renderer

# policy = Policy(diffusion, dataset.normalizer)

# #---------------------------------- main loop ----------------------------------#

# observation = env.reset()

# if args.conditional:
#     print('Resetting target')
#     env.set_target()

# ## set conditioning xy position to be the goal
# target = env._target
# cond = {
#     diffusion.horizon - 1: np.array([*target, 0, 0]),
# }

# ## observations for rendering
# rollout = [observation.copy()]
# total_reward = 0
# for t in range(env.max_episode_steps):

#     state = env.state_vector().copy()

#     ## can replan if desired, but the open-loop plans are good enough for maze2d
#     ## that we really only need to plan once
#     if t == 0:
#         cond[0] = observation
#         action, samples = policy(cond, batch_size=args.batch_size) # policy returns action, trajectories
#         actions = samples.actions[0] # 384
#         sequence = samples.observations[0] # 384 elements
#     # pdb.set_trace()

#     # If t is last index of sequence:
#     if t < len(sequence) - 1:
#         next_waypoint = sequence[t+1]
    
#     ## If we want to use calculated actions:
#     # --------------------------------------------------------------------------------------
#     else:
#         # Next waypoint is a copy of the last element of sequence
#         next_waypoint = sequence[-1].copy()
#         # Velocities in x and y are set to 0:
#         next_waypoint[2:] = 0
#         # pdb.set_trace()
            
#     # Can use actions or define a simple controller based on state predictions
#     # Action is defined as the positional difference + the velocity difference for x and y
#     action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:]) # Calculated actions
#     # --------------------------------------------------------------------------------------
#     # pdb.set_trace()
    

#     # Use actions defined in the process, instead of using next_waypoint
#     # --------------------------------------------------------------------------------------
#     # else:
#     #     # Actions are defined as the actions after the initial action, from index 1 and outwards.
#     #     actions = actions[1:]
#     #     if len(actions) > 1: # If actions only contains the first action after initial action, define as actions[0] (2nd action from policy)
#     #         action = actions[0] 
#     #     else: # Else action is defined as the negative value of state velocities.
#     #         action = -state[2:]
#     #--------------------------------------------------------------------------------------
    
#     # Section 4.3.1 Scoring refers to this:
#     next_observation, reward, terminal, _ = env.step(action)
#     total_reward += reward
#     score = env.get_normalized_score(total_reward)
#     print(
#         f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
#         f'{action}'
#     )

#     if 'maze2d' in args.dataset:
#         xy = next_observation[:2]
#         goal = env.unwrapped._target
#         print(
#             f'maze | pos: {xy} | goal: {goal}'
#         )

#     ## update rollout observations
#     rollout.append(next_observation.copy())

#     # logger.log(score=score, step=t)
#     if t % args.vis_freq == 0 or terminal:
#         fullpath = join(args.savepath, f'{t}.png')

#         if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)

#         # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

#         ## save rollout thus far
#         renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)
#         # renderer.composite(join(args.savepath, 'rollout_' + str(t) + '.png'), np.array(rollout)[None], ncol=1) # Makes the complete path(old rollout.png) now be the rollout_ + final t value + .png

#         # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

#         # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

#     if terminal:
#         break

#     observation = next_observation

# # save result as a json file
# json_path = join(args.savepath, 'rollout.json')
# json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
#     'epoch_diffusion': diffusion_experiment.epoch}
# json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)

# # logger.finish(t, env.max_episode_steps, score=score, value=0)

# def plot_observation_points(observations_to_plot, target):
#     observation_x = [point[0] for point in observations_to_plot]
#     observation_y = [point[1] for point in observations_to_plot]
#     plt.figure(figsize=(8, 6))
#     plt.plot(observation_x, observation_y, label="Observations", color='red', linewidth=1)
#     # Mark the first and last actions with points
#     plt.scatter(observation_x[0], observation_y[0], color='blue', label='Starting point', zorder=5)
#     plt.scatter(observation_x[-1], observation_y[-1], color='green', label='End point', zorder=5)
#     plt.scatter(target[0], target[1], color='gold', label='Target point', zorder=5)
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.title('Observation points', fontsize=14)
#     plt.xlabel('Observations_x', fontsize=12)
#     plt.ylabel('Observations_y', fontsize=12)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("plotted_observation_points.jpeg", format='jpeg', dpi=300)
