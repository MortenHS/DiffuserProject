import os
import numpy as np
import torch
import einops
import csv

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .cloud import sync_logs

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        n_samples=2,
        bucket=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema_model = self.model
        self.ema = EMA(ema_decay)
        
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        # --------------------------------------------Data Processing -----------------------------------------------#
        self.dataset = dataset

        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        
        # ------------------------------------------------------------------------------------------------------#

        # Tilsvarende til dataloader_vis i Diffuser?
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True))
        # ------------------------------------------------------------------------------------------------------#
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder

        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps, loss_log_file="loss_log.csv"): # for umaze = 10 000 = n_steps_per_epoch
        loss_data = []
        for step in range(n_train_steps):
            running_loss = 0.0
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)
                # Batch er delt opp i: Trajectories [batch_size=32, horizon, dim=6], og
                # conditions: {{0: tensor([[-0.5100,  0.0400,  0.0019,  0.0042]], device='cuda:0'), 
                #     127: tensor([[ 0.6872,  0.8385, -0.7158,  0.0234]], device='cuda:0')}} for CFM

                loss, infos = self.model.loss(*batch) # In CFM/GaussianDiffusion
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                running_loss += loss.item()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples(n_samples=self.n_samples)

            loss_data.append([self.step, running_loss])
            self.step += 1
        
        with open(loss_log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if step == 0:
                writer.writerow(["Step", "Loss"])
            writer.writerows(loss_data)

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath, weights_only=True)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def load_model(self, loadpath):
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''  
        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
        
        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders n_samples samples from generative model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            # repeat each item in conditions `n_samples` times: conditions[0].shape goes from [1, 4] to [10, 4]
            if type(self.ema_model).__name__ == "GaussianDiffusion":
                conditions = apply_dict(
                    einops.repeat,
                    conditions,
                    'b d -> (repeat b) d', repeat=n_samples,
                ) 
                
                # [ n_samples x horizon x (action_dim + observation_dim) ]
                samples = self.ema_model.conditional_sample(conditions)
                samples = to_np(samples)

                # [ n_samples x horizon x observation_dim ]
                normed_observations = samples[:, :, self.dataset.action_dim:] # [32, 128, 4]
                
                # [ 1 x 1 x observation_dim ]
                normed_conditions = to_np(batch.conditions[0])[:, None] # [1, 1, 4]
                
                # [ n_samples x (horizon + 1) x observation_dim ]
                normed_observations = np.concatenate([
                    np.repeat(normed_conditions, n_samples, axis=0),
                    normed_observations
                ], axis=1)

                # Shape of normed_observations: [10, 129, 4]
                # [ n_samples x (horizon + 1) x observation_dim ]
                observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
                savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
                self.renderer.composite(savepath, observations, plot_goal=False)

            else:
                # Conditions[0] has shape [1, 4]
                conditions = apply_dict(
                    einops.repeat,
                    conditions,
                    'b d -> (repeat b) d', repeat=n_samples,
                )
                # conditions[0].shape = [10, 4]

                # [ n_samples x horizon x (action_dim + observation_dim) ]
                samples = self.ema_model.conditional_sample(conditions) # conditions er 2 ganger [10,4]
                samples = to_np(samples)
                # samples shape: [32, 128, 6], 6 = 4+2 action_dim + observation_dim

                # [ n_samples x horizon x observation_dim ]
                normed_observations = samples[:, :, self.dataset.action_dim:] # [32, 128, 4]

                                # [ 1 x 1 x observation_dim ]
                normed_conditions = to_np(batch.conditions[0])[:, None] # [1, 1, 4]
                
                # [ n_samples x (horizon + 1) x observation_dim ]
                normed_observations = np.concatenate([
                    np.repeat(normed_conditions, n_samples, axis=0),
                    normed_observations
                ], axis=1)
                
                # Shape of normed_observations: [1, 128, 4]
                observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
                savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
                self.renderer.composite(savepath, observations, plot_goal=False)
    
