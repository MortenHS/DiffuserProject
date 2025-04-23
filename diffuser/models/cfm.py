import numpy as np
import torch
from torch import nn
import pdb
import copy
from torchcfm.conditional_flow_matching import *
from torchdyn.core import NeuralODE
import torchdiffeq

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

class CFM(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, use_wavelet=False
    ):
        super().__init__()
        if use_wavelet:
            self.horizon = horizon // 2
        else:
            self.horizon = horizon

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        sigma = 0.0
        # self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        self.FM = ConditionalFlowMatcher(sigma=sigma)
        self.node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.betas = betas
        # self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        # self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1- alphas_cumprod)

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        # self.register_buffer('betas', betas)
        # self.register_buffer('alphas_cumprod', alphas_cumprod)
        # self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # # calculations for diffusion q(x_t | x_{t-1}) and others
        # self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        # self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        # self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        # self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # # calculations for posterior q(x_{t-1} | x_t, x_0)
        # posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # self.register_buffer('posterior_variance', posterior_variance)

        # ## log calculation clipped because the posterior variance
        # ## is 0 at the beginning of the diffusion chain
        # self.register_buffer('posterior_log_variance_clipped',
        #     torch.log(torch.clamp(posterior_variance, min=1e-20)))
        # self.register_buffer('posterior_mean_coef1',
        #     betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        # self.register_buffer('posterior_mean_coef2',
        #     (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        # ## get loss coefficients and initialize objective
        # loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def set_sampling_timesteps(self, t):
        self.n_timesteps = t

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    # ------------------------------------------Sampling------------------------------------------#
    def predict_start_from_noise(self, x_t, t, noise):
        '''
            Reconstructs original data from noisy sample at timestep t

            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise
    

    def p_mean_variance(self, x, cond, t):
        ''' 
        Performs one denoising step in the reverse process
        '''
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance


    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, global_cond=None)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop_original(self, shape, cond, verbose=True, return_diffusion=False):
        device = self.betas.device
        print(f"\n Cond in p_sample_loop_original: {cond}\n")
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, global_cond, cond, timesteps)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def p_sample_loop_cfm(self, shape, cond, verbose=True, return_diffusion=False):
        # x shape here: [32, 128, 6] == [B, horizon, dim]
        # t shape here: [] !! Problem
        traj = torchdiffeq.odeint(
            lambda t, x: (self.model.forward(x, cond, time=t.expand(x.shape[0]))),

            # # Lambda har [1] til slutt fordi den blir en tuple med print statement.
            # lambda t, x: (print(f"\nx shape in odeint func: {x.shape}, t shape: {t.shape}, t value: {t}")
            # , self.model.forward(x, cond, time=t.expand(x.shape[0])))[1],

            torch.randn(shape).to(self.device),
            torch.linspace(0, 1, self.n_timesteps + 1).to(self.device),
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
    
        return traj[-1]


    def p_sample_loop(self, shape, cond, verbose=True, return_diffusion=False, **kwargs):
        sample_type = kwargs.get('sample_type', 'original')

        return self.p_sample_loop_cfm(shape, cond, verbose, return_diffusion)
        # return self.p_sample_loop_original(shape, verbose, return_diffusion)
    

        # if sample_type == 'repaint':
        #     return self.p_sample_loop_repaint(shape, global_cond, verbose, return_diffusion)
        # elif sample_type == 'constrained':
        #     return self.p_sample_loop_constrained(shape, global_cond, verbose, return_diffusion)
        # elif sample_type == 'original':
        #     return self.p_sample_loop_original(shape, global_cond, verbose, return_diffusion)
        # elif sample_type == 'estimate_feature':
        #     return self.p_sample_loop_estimate_feature(shape, global_cond, verbose, return_diffusion)
        # else:
        #     raise NotImplementedError

    def conditional_sample(self, cond, *args, horizon=None, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.device
        batch_size = 32
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        # global_cond = global_cond.to(device)
        # global_cond = {k: v.to(device) for k, v in global_cond.items()}
        # for k, v in global_cond.items():
        #     if type(v) is torch.Tensor:
        #         global_cond[k] = v.to(device)

        return self.p_sample_loop(shape, cond, *args, **kwargs)


    #------------------------------------------ training ------------------------------------------#
    @property
    def device(self):
        """Get the device where the model's parameters are allocated."""
        # Assumes the model's parameters are all on the same device.
        return next(self.parameters()).device

    def loss(self, x, cond): # def loss(self, x, global_cond, cond):
        ''' 
        Hvis implementasjon i CondUnet1D, sørg for å legge til global_cond=None
        '''
        x = x.to(self.device)
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        
        x1 = x.to(self.device)
        x0 = torch.randn_like(x1)
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)

        # I T-CFM: vt = self.model(t, xt, global_cond=global_cond)

        vt = self.model(xt, cond, t)
        loss = torch.mean((vt - ut) ** 2)
        return loss, {'loss': loss.item()}


    def forward(self, *args, **kwargs):
        return self.conditional_sample(*args, **kwargs)


    # def loss(self, x, cond):
    #     batch_size = len(x)
    #     t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()

    #     print(f"\nLoss input information, batch size: {batch_size}, t: {t.shape}, cond: {cond}")

    #     return self.p_losses(x, cond, t)

    # def q_sample(self, x_start, t, noise=None):
    #     if noise is None:
    #         noise = torch.randn_like(x_start)

    #     device = self.device
    #     self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
    #     self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
    #     sample = (
    #         extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
    #         extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    #     )

    #     print(f"Noise: {noise.shape}")
    #     return sample

    # def p_losses(self, x_start, cond, t):
    #     noise = torch.randn_like(x_start)

    #     x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    #     x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

    #     x_recon = self.model(x_noisy, cond, t)
    #     x_recon = apply_conditioning(x_recon, cond, self.action_dim)

    #     assert noise.shape == x_recon.shape

    #     if self.predict_epsilon:
    #         loss, info = self.loss_fn(x_recon, noise)
    #     else:
    #         loss, info = self.loss_fn(x_recon, x_start)

    #     print(f"Loss and info shapes: loss: {loss.shape}, info: {info.shape}")
    #     return loss, info