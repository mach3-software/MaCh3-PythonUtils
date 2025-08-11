from typing import Any, Callable

import torch

class VAELoss():
    def __call__(self, x: torch.Tensor, x_hat: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor, generic_loss: Callable) -> Any:
        """
        Compute the loss for a Variational Autoencoder (VAE).
        
        :param x: Original input tensor
        :param x_hat: Reconstructed output tensor
        :param mean: Mean of the latent space distribution
        :param log_var: Log variance of the latent space distribution
        :return: Total VAE loss (reconstruction loss + KL divergence)
        """
        # Reconstruction loss (Mean Squared Error)
        recon_loss = generic_loss(x_hat, x, reduction='sum')

        # KL Divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        
        return recon_loss + kl_div