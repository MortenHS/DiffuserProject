import numpy as np
from os.path import join

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'

def plan_maze(args):
    """
    Executes the maze planning logic and returns the final score and reward.
    Args:
        args: Parsed arguments from the Parser class.
    """
    # Setup
    env = datasets.load_environment(args.dataset)
    diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, 
        args.diffusion_loadpath, epoch=args.diffusion_epoch,
    )
    diffusion = diffusion_experiment.ema
    dataset_obj = diffusion_experiment.dataset
    renderer = diffusion_experiment.renderer
    policy = Policy(diffusion, dataset_obj.normalizer)

    # Initialize environment
    observation = env.reset()
    if args.conditional:
        env.set_target()
    target = env._target

    method_name = "cfm" if args.config.endswith("_cfm") else "diff"
    cond = {diffusion.horizon - 1: np.array([*target, 0, 0])}

    # Main loop
    rollout = [observation.copy()]
    total_reward = 0
    for t in range(env.max_episode_steps):
        state = env.state_vector().copy()

        if t == 0:
            cond[0] = observation
            action, samples = policy(cond, batch_size=args.batch_size)
            actions = samples.actions[0]
            sequence = samples.observations[0]

        if t < len(sequence) - 1:
            next_waypoint = sequence[t + 1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0

        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        score = env.get_normalized_score(total_reward)

        if terminal:
            break

        observation = next_observation
        rollout.append(next_observation.copy())

        if t % args.vis_freq == 0 or terminal:
            renderer.composite(
                join(args.savepath, f"rollout_{method_name}.png"), np.array(rollout)[None], ncol=1
            )

    return score, total_reward