from collections import namedtuple
import torch
import einops
import diffuser.utils as utils

Trajectories = namedtuple('Trajectories', 'actions observations')

class Policy:
    def __init__(self, model, normalizer):
        self.model = model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim

    @property
    def device(self):
        parameters = list(self.model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        """
        Formats and prepares the input `conditions` dictionary for model processing.

        This function performs the following steps:
        1. Normalizes the 'observations' entry in the `conditions` dictionary using the provided normalizer.
        2. Converts all entries in the `conditions` dictionary to PyTorch tensors with dtype float32 and moves them to the CUDA device 'cuda:0'.
        3. Repeats each tensor in the `conditions` dictionary along a new batch dimension to match the specified `batch_size`.

        Args:
            conditions (dict): A dictionary containing condition data
            batch_size (int): The number of times to repeat each condition to match the batch size required by the model.

        Returns:
            dict: A dictionary with the same keys as `conditions`, where each value is a normalized, CUDA tensor repeated along the batch dimension.
        """
        # Step 1:
        conditions = utils.apply_dict( 
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        # Step 2:
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')

        # Step 3:
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def __call__(self, conditions, debug=False, batch_size=1):
        conditions = self._format_conditions(conditions, batch_size)

        # Calls forward for given model: run reverse diffusion process, run conditional_sample function
        sample = self.model(conditions) # Calls forward in CFM or diffusion.py
        sample = utils.to_np(sample)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        action = actions[0, 0]

        ## extract observations [ batch_size x horizon x observation_dim ]
        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
    
        trajectories = Trajectories(actions, observations)
        return action, trajectories