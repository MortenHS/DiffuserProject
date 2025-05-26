import json
import numpy as np
from os.path import join

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

#-------------------------------------------------------------------- setup --------------------------------------------------------------------#
class Parser(utils.Parser):
    dataset: str = 'maze2d-medium-v1'
    config: str = 'config.maze2d_cfm'
    sampling_steps: int = 1

args = Parser().parse_args('plan')
env = datasets.load_environment(args.dataset)

#-------------------------------------------------------------------- Load model specifics --------------------------------------------------------------------#
args.logbase = '/cluster/work/mortenhs/Janner/diffuser/logs/'
diffusion_experiment = utils.load_diffusion(
    args.logbase, 
    args.dataset, 
    args.diffusion_loadpath, 
    epoch=488800 # 520000.pt, args.diffusion_epoch
    ) 
sampling_steps = args.sampling_steps

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

if args.config.endswith('_cfm'): method_name = 'cfm'
else: method_name = 'diff'


policy = Policy(diffusion, sampling_steps, dataset.normalizer)
#-------------------------------------------------------------------- main planning loop --------------------------------------------------------------------#
observation = env.reset()

# Single vs multi-task? Single = False, Multi = True
if args.conditional:
    print('Resetting target')
    env.set_target()

# set conditioning xy_pos position to be the goal
target = env._target

cond = {
    0 : observation, # Shape (4,)
    diffusion.horizon - 1: np.array([*target, 0, 0]), # [1 1 0 0], [6, 6, 0, 0], [7, 9, 0, 0]
}

action, samples = policy(cond, batch_size=args.batch_size)
actions = samples.actions[0] # (128, 2)
sequence = samples.observations[0] # (Horizon: 128, observation_dim: 4), len = 128     

rollout = [observation.copy()]
total_reward = 0
for t in range(env.max_episode_steps):
    state = env.state_vector().copy()
    if t == 0:
        initial_state = state.copy()

    # While trajectory length is not reached, run the controller
    if t < len(sequence) - 1:
        next_waypoint = sequence[t+1]

    else: # When trajectory length is reached, stop the controller
        next_waypoint = sequence[-1].copy()
        # Velocities in x and y are set to 0:
        next_waypoint[2:] = 0
            
    # Can use actions or define a simple controller based on state predictions
    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
    
    next_observation, reward, terminal, _ = env.step(action)
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'action : {action}'
    )

    if 'maze2d' in args.dataset:
        xy_pos = next_observation[:2]
        goal = env.unwrapped._target
        # print(
        #     f'maze | pos: {xy_pos} | goal: {goal}'
        # )
    
#----------------------------------------------------------------------- Rendering and saving plots --------------------------------------------------------------------#

    # # update rollout observations
    # rollout.append(next_observation.copy())

    # if t == 0:
    #     fullpath = join(args.savepath, f'0_{method_name}_N_{sampling_steps}.png')
    #     renderer.composite(fullpath, samples.observations, ncol=1, plot_goal=True, goal=goal)

    # # if t % 100 == 0:
    # #     renderer.composite(join(args.savepath, f'rollout_{method_name}_{t}.png'), np.array(rollout)[None], ncol=1, plot_goal=True, goal=goal)
    # if t == env.max_episode_steps - 1:
    #     renderer.composite(join(args.savepath, f'rollout_{method_name}_N_{sampling_steps}.png'), np.array(rollout)[None], ncol=1, plot_goal=True, goal=goal)

    # observation = next_observation

# print(f"Initial state: {initial_state}")
# print(f"Final state: {state}")
# print(f"Conditions: Initial {cond[0]} | Final {cond[diffusion.horizon - 1]}")
# print(f"First position of samples.observations: {samples.observations[0][0]}") # First element in sequence
# print(f"First position of rollout: {rollout[0]}")
# print(f"Final position of samples.observations: {samples.observations[0][-1]}") # Last element in sequence
# print(f"Final position of rollout: {rollout[-1]}")


#---------------------------------- Save to JSON file ---------------------------------------------------------------------------------------#
# json_path = join(args.savepath, f'rollout_{method_name}_N_{sampling_steps}.json')
# json_data = {
#     'score': score, 
#     'step': t, 
#     'return': total_reward, 
#     'term': terminal,
#     'epoch_diffusion': diffusion_experiment.epoch}

# json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
# print(f"Json saved to {json_path}")