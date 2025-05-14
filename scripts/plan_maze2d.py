import json
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d_cfm'

#---------------------------------- setup ----------------------------------#
args = Parser().parse_args('plan')

env = datasets.load_environment(args.dataset)
#---------------------------------- loading ----------------------------------#
args.logbase = '/cluster/work/mortenhs/Janner/diffuser/logs/tests_2'
diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=940) # 520000.pt, args.diffusion_epoch
# print(f"Loading diffusion from: {join(args.logbase, args.dataset, args.diffusion_loadpath)}")
# logs, maze2d-dataset-v1, cfm/H128_T64

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

#---------------------------------- main loop ----------------------------------#
observation = env.reset()

# def set_target(self, target_location=None):
#     if target_location is None:
#         idx = self.np_random.choice(len(self.empty_and_goal_locations))
#         reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
#         target_location = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
#     self._target = target_location

# Single vs multi-task? Single = False, Multi = True
if args.conditional:
    print('Resetting target')
    env.set_target()

# set conditioning xy position to be the goal
target = env._target

if args.config.endswith('_cfm'): method_name = 'cfm'
else: method_name = 'diff'

cond = {
    diffusion.horizon - 1: np.array([*target, 0, 0]),
}

# observations for rendering
rollout = [observation.copy()]
total_reward = 0
for t in range(env.max_episode_steps):

    state = env.state_vector().copy()

    # can replan if desired, but the open-loop plans are good enough for maze2d
    # that we really only need to plan once
    if t == 0:
        cond[0] = observation # Shape (4,)
        action, samples = policy(cond, batch_size=args.batch_size) # policy returns action, trajectories
        actions = samples.actions[0]
        sequence = samples.observations[0]

    # If t is last index of sequence:
    if t < len(sequence) - 1:
        next_waypoint = sequence[t+1]
    
    # If we want to use calculated actions:
    # --------------------------------------------------------------------------------------
    else:
        # Next waypoint is a copy of the last element of sequence
        next_waypoint = sequence[-1].copy()
        # Velocities in x and y are set to 0:
        next_waypoint[2:] = 0
            
    # Can use actions or define a simple controller based on state predictions
    # Action is defined as the positional difference + the velocity difference for x and y
    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:]) # Calculated actions

    # --------------------------------------------------------------------------------------
    
    # Use actions defined in the process, instead of using next_waypoint
    # --------------------------------------------------------------------------------------
    # else:
    #     # Actions are defined as the actions after the initial action, from index 1 and outwards.
    #     actions = actions[1:]
    #     if len(actions) > 1: # If actions only contains the first action after initial action, define as actions[0] (2nd action from policy)
    #         action = actions[0] 
    #     else: # Else action is defined as the negative value of state velocities.
    #         action = -state[2:]
    #--------------------------------------------------------------------------------------
    
    # Section on Scoring refers to this:
    next_observation, reward, terminal, _ = env.step(action)
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'action : {action}'
    )

    if 'maze2d' in args.dataset:
        xy = next_observation[:2]
        goal = env.unwrapped._target
        print(
            f'maze | pos: {xy} | goal: {goal}'
        )

    # update rollout observations
    rollout.append(next_observation.copy())

    if t % args.vis_freq == 0 or terminal:
        fullpath = join(args.savepath, f'{t}_{method_name}.png')

        if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)
            
        ## save rollout thus far
        renderer.composite(join(args.savepath, f'rollout_{method_name}.png'), np.array(rollout)[None], ncol=1)

    if terminal:
        break

    observation = next_observation


# save result as a json file
json_path = join(args.savepath, f'rollout_{method_name}_tests_2.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
print(f"Json saved to {json_path}")