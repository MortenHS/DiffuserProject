from collections import namedtuple
import torch
import einops
import diffuser.utils as utils

Trajectories = namedtuple('Trajectories', 'actions observations')

class Policy:
    def __init__(self, diffusion_model, normalizer):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def __call__(self, conditions, debug=False, batch_size=1):
        print(f"Conds before_format: {conditions[0]}")
        conditions = self._format_conditions(conditions, batch_size)
        print(f"Conds after_format: {conditions[0]}")
        # Calls forward for given model: run reverse diffusion process, run cond_sample 
        sample = self.diffusion_model(conditions) # Calls forward in CFM or diffusion.py
        sample = utils.to_np(sample)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        action = actions[0, 0]

        # if debug:
        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
    
        trajectories = Trajectories(actions, observations)
        return action, trajectories

class PolicyFM:
    def __init__(self, flow_model, normalizer):
        self.flow_model = flow_model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim

    @property
    def device(self):
        parameters = list(self.flow_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def __call__(self, conditions, debug=False, batch_size=1):
        print(f"Conds before_format: {conditions[0]}")
        conditions = self._format_conditions(conditions, batch_size)
        print(f"Conds after_format: {conditions}")

        # For flow matching, you typically sample by integrating the learned vector field
        # This may look like: sample = self.flow_model.sample(conditions)
        # The sample shape should be [batch_size, horizon, obs_dim + action_dim]
        sample = self.flow_model.conditional_sample(conditions)
        sample = utils.to_np(sample)

        # Extract actions and observations as before
        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        action = actions[0, 0]

        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories(actions, observations)
        return action, trajectories