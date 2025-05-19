import torch
from torch import nn
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdyn.core import NeuralODE
import torchdiffeq

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
        self.model_type = model.__class__.__name__ # TempUnet or Cu1D

        sigma = 0.0
        self.FM = ConditionalFlowMatcher(sigma=sigma)
        self.node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.betas = betas

        self.n_timesteps = int(n_timesteps) # For umaze = 64
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
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
    # def p_sample_loop_cfm(self, shape, cond, verbose=True, return_diffusion=False):
    #     # x shape here: [32, 128, 6] == [B, horizon, dim]
    #     overwritten_timesteps = 256
    #     if self.model_type == 'ConditionalUnet1D':
    #         traj = torchdiffeq.odeint(
    #             lambda t, x: (self.model.forward(t=t.expand(x.shape[0]), x=x, global_cond=cond)),
    #             torch.randn(shape).to(self.device),
    #             torch.linspace(0, 1, overwritten_timesteps + 1).to(self.device),
    #             atol=1e-4,
    #             rtol=1e-4,
    #             method="euler",
    #         )
    #         return traj[-1]
    #     else:
    #         raise ValueError(f"Unsupported model type: {self.model_type}")

    def p_sample_loop_cfm(self, shape, cond, verbose=True, return_diffusion=False):
        if self.model_type == 'ConditionalUnet1D':
            traj = torchdiffeq.odeint(
                lambda t, x: self.model.forward(
                    t=t.expand(x.shape[0]), 
                    x=apply_conditioning(x, cond, self.action_dim), 
                    global_cond=None
                ),
                torch.randn(shape).to(self.device),
                torch.linspace(0, 1, self.n_timesteps + 1).to(self.device),
                atol=1e-4,
                rtol=1e-4,
                method="euler",
            )

            # Before/without apply cond: [65, 1, 128, 6], traj[-1]: [1, 128, 6]
            # After apply cond: [1, 128, 6], traj[-1]: [128, 6]

            # traj[-1] = apply_conditioning(traj[-1], cond, self.action_dim)
            # print(f"Cond[0][0]: {cond[0][0]}")
            # print(f"Traj[-1][0] after applying conditioning: {traj[-1][0]}")
            return traj[-1]
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")


    def p_sample_loop(self, shape, cond, verbose=True, return_diffusion=False, **kwargs):
        return self.p_sample_loop_cfm(shape, cond, verbose, return_diffusion)
    

    def conditional_sample(self, cond, *args, horizon=None, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.device
        batch_size = len(cond[0]) # 1 for CFM
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, *args, **kwargs)


    #------------------------------------------ training ------------------------------------------#
    @property
    def device(self):
        """
        Get the device where the model's parameters are allocated,
        assuming all parameters are on the same device.
        """
        return next(self.parameters()).device

    def loss(self, x, cond):
        x = x.to(self.device)
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        
        x1 = x.to(self.device)
        x0 = torch.randn_like(x1)
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)

        # Apply conditioning to xt for inpainting purposes.
        xt = apply_conditioning(xt, cond, self.action_dim)

        if self.model_type == 'ConditionalUnet1D':
            vt = self.model(t, xt, global_cond=None) 
            vt = apply_conditioning(vt, cond, self.action_dim)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        loss = torch.mean((vt - ut) ** 2)
        return loss, {'loss': loss.item()}

    def forward(self, *args, **kwargs):
        return self.conditional_sample(*args, **kwargs)
