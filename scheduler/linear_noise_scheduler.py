import torch
import numpy as np


class LinearNoiseScheduler:
    r"""
    Class for the linear noise scheduler that is used in DDPM.
    """
    
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        # Mimicking how compvis repo creates schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
            #torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2

        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    
    def add_noise(self, original, noise, t):

        original_shape = original.shape
        batch_size = original_shape[0]

        # select sqrt_alpha_cum_prod for the current timesteps
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        # select sqrt_one_minus_alpha_cum_prod for the current timesteps
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        
        # Reshape the above two with (B,) shape to (B,1,1,1) to match the dimension of latents (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
        
        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)
    
    def sample_prev_timestep(self, xt, noise_pred, t):

        x0 = ((xt - (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred)) /
              torch.sqrt(self.alpha_cum_prod.to(xt.device)[t]))
        # x0 = torch.clamp(x0, 0, 1)
        
        mean = xt - ((self.betas.to(xt.device)[t]) * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t])
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])
        
        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]) / (1.0 - self.alpha_cum_prod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)

            return mean + sigma * z, x0

    def training_reverse_diffuse(self, xt, noise_pred, t):
        B = xt.shape[0]

        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t].view(B, 1, 1, 1)
        alpha = self.alpha_cum_prod.to(xt.device)[t].view(B, 1, 1, 1)

        x0 = (xt - sqrt_one_minus_alpha * noise_pred) / torch.sqrt(alpha)
        return x0
