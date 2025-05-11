import diffuser.utils as utils
import diffuser.datasets as datasets
from os.path import join
from diffuser.guides.policies import Policy
import numpy as np
import imageio
# Conditional Sampling
class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

args = Parser().parse_args('plan')

diffusion_experiment = utils.load_diffusion(
    args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

dataset = diffusion_experiment.dataset
diffusion = diffusion_experiment.ema
renderer = diffusion_experiment.renderer
model = diffusion_experiment.trainer.ema_model

policy = Policy(diffusion, dataset.normalizer)

env = datasets.load_environment(args.dataset)
observation = env.reset()

obs = utils.colab.run_diffusion(model, dataset, observation, n_samples=10)

# Conditioning on target (final) 
target = env._target
cond = {
    diffusion.horizon - 1: np.array([*target, 0, 0]),
}

# Use final observations for evaluation/visualization
final = obs[-1]    # [256 x horizon x obs_dim]
total_reward = 0
state = env.state_vector().copy()
cond[0] = obs

print(f"Cond shape {cond[0].shape}") # (2, 10, 384, 10)
action, samples = policy(cond, batch_size=args.batch_size) # policy returns action, trajectories
actions = samples.actions[0]
sequence = samples.observations[0]

next_waypoint = sequence[-1].copy()
next_waypoint[2:] = 0

action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

next_observation, reward, terminal, _ = env.step(action)
total_reward += reward
score = env.get_normalized_score(total_reward)

img = renderer.renders(final[0, :, :2])  # pick first trajectory
imageio.imwrite('logs/trajectory_sample.png', img)

# Gir unexpected keyword argument in p_sample_loop(). Kan muligens hardkodes?

# n_samples = 1
# observations = utils.colab.run_diffusion(
#     model, dataset, observation, n_samples, args.device)

# # observations = utils.colab.run_diffusion(
# #     model, dataset, observation, args.n_diffusion_steps, args.device)

# # print(observations.shape) # (65, 64, 128, 4)
# # print(args.n_diffusion_steps) # 64

# final_obs = observations[-1]       # shape: [n_samples, horizon, obs_dim]
# positions = final_obs[:, :, :2]    # shape: [n_samples, horizon, 2]
# print(f"Len positions: {len(positions)}")

# if len(positions)==256:
#     ncol = 16
# elif len(positions)==1:
#     ncol = 1

# renderer.composite(f'logs/sample_image_{n_diffusion_steps}.png', paths=positions, ncol=ncol)