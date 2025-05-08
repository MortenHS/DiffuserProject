import diffuser.utils as utils
import diffuser.datasets as datasets
from os.path import join
import numpy as np

# Conditional Sampling
class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

args = Parser().parse_args('plan')

diffusion_experiment = utils.load_diffusion(
    args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer
model = diffusion_experiment.trainer.ema_model

env = datasets.load_environment(args.dataset)
observation = env.reset()

# observations = utils.colab.run_diffusion(
#     model, dataset, observation, 256, args.device)

observations = utils.colab.run_diffusion(
    model, dataset, observation, args.n_diffusion_steps, args.device)
# utils.colab.show_diffusion(renderer, observations[:, :5], substep=1)

# print(observations.shape) # (65, 64, 128, 4)
# print(args.n_diffusion_steps) # 64

final_obs = observations[-1]       # shape: [n_samples, horizon, obs_dim]
positions = final_obs[:, :, :2]    # shape: [n_samples, horizon, 2]
# print(f"Len positions: {len(positions)}")


# renderer.composite('logs/sample_image.png', paths=positions, ncol=5)
