from typing import Optional, List

import torch

class TorchPerceptronModel(torch.nn.Module):
    """
    Base class for PyTorch neural networks.
    Inherits from torch.nn.Module to ensure compatibility with PyTorch's training and evaluation routines.
    """
    def __init__(self, input_dim: int, output_dim: int, layers: Optional[List[torch.nn.Module]] = None):
        torch.nn.Module.__init__(self)

        self.input_dim = input_dim
        self.output_dim = output_dim

        # check dimensions of the first and last layers
        if layers is not None:
            if layers[0].in_features != input_dim:
                raise ValueError(f"First layer input dimension {layers[0].in_features} does not match model input dimension {input_dim}")
            if layers[-1].out_features != output_dim:
                raise ValueError(f"Last layer output dimension {layers[-1].out_features} does not match model output dimension {output_dim}")
        
        self._interface = torch.nn.Sequential()
        if layers:
            for layer in layers:
                self.add_layer(layer)        

    def add_layer(self, layer: torch.nn.Module):
        """
        Add a layer to the network.
        
        :param layer: A PyTorch layer (e.g., torch.nn.Linear, torch.nn.Conv2d, etc.)
        """
        if not isinstance(layer, torch.nn.Module):
            raise TypeError("Layer must be an instance of torch.nn.Module")
        self._interface.append(layer)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        :param x: Input tensor
        :return: Output tensor after passing through the network
        """
        return self._interface(x)


class TorchVAEModel(torch.nn.Module):
    """
    Base class for Variational Autoencoders (VAEs) in PyTorch.
    Inherits from torch.nn.Module to ensure compatibility with PyTorch's training and evaluation routines.
    """
    def __init__(self, encoder: TorchPerceptronModel, decoder: TorchPerceptronModel):
        torch.nn.Module.__init__(self)

        self.encoder = encoder
        self.decoder = decoder
        
        self.mean_dim = torch.nn.Linear(encoder.output_dim, 2)  # Mean and log variance
        self.log_var_dim = torch.nn.Linear(encoder.output_dim, 2)  #
        if self.decoder.input_dim != 2:
            raise ValueError("Decoder input dimension must be 2 for VAE (mean and log variance)")
        
        if self.encoder.input_dim != self.decoder.output_dim:
            raise ValueError("Encoder output dimension must match decoder input dimension for VAE")
        
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the input tensor into a latent space representation.
        
        :param x: Input tensor
        :return: Latent space representation
        """
        encoded = self.encoder(x)
        mean = self.mean_dim(encoded)
        log_var = self.log_var_dim(encoded)
        return mean, log_var
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent space representation back to the original space.
        
        :param z: Latent space representation
        :return: Reconstructed tensor
        """
        return self.decoder(z)
    
    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from the latent space.
        
        :param mean: Mean of the latent space distribution
        :param log_var: Log variance of the latent space distribution
        :return: Sampled tensor from the latent space
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        :param x: Input tensor
        :return: Reconstructed tensor
        """
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var