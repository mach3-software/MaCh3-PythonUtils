import numpy as np
from typing import Callable, Tuple

import torch
from tqdm.auto import tqdm

from mach3pythonutils.ml_algorithms.interfaces.ml_interface import MLAlgorithm, MLInterface
from mach3pythonutils.file_io.root_dataset import ROOTDataset
from mach3pythonutils.utils.utils import (
    get_styled_logger, 
    log_section_header, 
    log_success, 
    log_info,
    log_debug
)
from mach3pythonutils.ml_algorithms.implementations.torch.custom_loss_functions import VAELoss

class TorchNNAlgorithm(MLAlgorithm, torch.nn.Module):
    def __init__(self, algorithm: torch.nn.Module):
        torch.nn.Module.__init__(self)
        MLAlgorithm.__init__(self, algorithm)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._algorithm.to(self.device)

    def train(self, dataset: ROOTDataset, loss_func: Callable = torch.nn.MSELoss(), **kwargs):
        """
        Define the training loop for the algorithm.
        :param dataset: The dataset to train on.
        :type dataset: ROOTDataset
        :param loss_func: The loss function to use for training.
        :type loss_func: torch.nn.modules.loss._Loss
        :param kwargs: Additional training parameters such as learning rate, epochs, etc.
        """
        # Get a styled logger for this class
        logger = get_styled_logger(self.__class__.__name__)

        early_stopping = kwargs.get('early_stopping', False)
        n_early_stopping = kwargs.get('n_early_stopping', 10)
        epochs = kwargs.get('epochs', 10)
        learning_rate = kwargs.get('learning_rate', 0.001)
        
        if not isinstance(dataset, ROOTDataset):
            raise TypeError("dataset must be an instance of ROOTDataset")
        
        # Log training configuration
        log_section_header(logger, "PyTorch Training Configuration")
        log_debug(logger, f"Epochs: [yellow]{epochs}[/yellow]")
        log_debug(logger, f"Learning Rate: [yellow]{learning_rate}[/yellow]")
        log_debug(logger, f"Loss Function: [yellow]{loss_func.__class__.__name__}[/yellow]")
        log_debug(logger, f"Early Stopping: [yellow]{early_stopping}[/yellow]")
        if early_stopping:
            log_info(logger, f"Early Stopping Patience: [yellow]{n_early_stopping}[/yellow]")
        
        self._algorithm.train()
        optimizer = torch.optim.Adam(self._algorithm.parameters(), lr=learning_rate)
        criterion = torch.nn.functional.mse_loss

        log_section_header(logger, "Training Progress")
        
        self._loss_values = np.zeros(epochs)

        data, labels = dataset.get_items(0, len(dataset))
        data = data.to(self.device)
        labels = labels.to(self.device)
        print(data)

        for epoch in (pbar:=tqdm(range(epochs))):
            # epoch_loss = self.training_step(optimizer, data, labels, criterion)
            optimizer.zero_grad()
            outputs = self._algorithm(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate average loss for the epoch
            self._loss_values[epoch] = loss
            # Log progress with rich formatting
                
            if early_stopping and self.early_stopping(n_early_stopping):
                log_info(logger, "[bold orange3]Early stopping triggered![/bold orange3]")
                break
            
            pbar.set_description(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.5f}")
            
            # In case the loading bar breaks
            if kwargs.get("print_step", False) and (epoch + 1):                
                # Print the loss for the current epoch every 10% of the training
                if (epoch + 1) % (epochs // 10) == 0 or epoch == epochs - 1:
                    log_info(logger, f"Epoch {epoch+1} //  - Loss: {loss:.5f}")
            

        log_success(logger, f"Training completed! Final loss: {self.get_loss_values()[-1]:.6f} | Total epochs: {len(self.get_loss_values())}")
    
    def training_step(self, optimizer, data, labels, criterion):
        # Convert data and labels to the appropriate tensors
        data, labels = data.to(self.device), labels.to(self.device)
        optimizer.zero_grad()
        outputs = self._algorithm(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def get_loss_values(self)->np.ndarray:
        """
        Get the loss values recorded during training.
        
        :return: Array of loss values.
        :rtype: np.ndarray
        """
        return self._loss_values[self._loss_values > 0]

    def early_stopping(self, patience: int = 10):
        """
        Check if early stopping criteria are met.
        
        :param patience: Number of epochs to wait before stopping if no improvement.
        :type patience: int
        :return: True if early stopping criteria are met, False otherwise.
        :rtype: bool
        """        
        logger = get_styled_logger(self.__class__.__name__)
        
        if len(self.get_loss_values()) < patience:
            return False
        
        if np.all(np.diff(self.get_loss_values()[-patience:]) >= 0):
            log_info(logger, "[bold orange3]⏹️  Early stopping triggered - no improvement detected[/bold orange3]")
            return True
        
        return False
    
    def predict(self, dataset: ROOTDataset) -> torch.Tensor:
        """
        Get algorithm to predict labels for the given dataset.
        
        :param dataset: The dataset to predict on.
        :type dataset: ROOTDataset
        :return: Predicted labels.
        :rtype: torch.Tensor
        """
        logger = get_styled_logger(self.__class__.__name__)
        log_info(logger, "🔮 Running predictions on dataset...")
        
        self._algorithm.eval()
        data, _ = dataset.get_items(0, len(dataset))
        
        with torch.no_grad():
            predictions = self._algorithm(data)
            
        log_success(logger, f"Predictions completed!")
        return predictions
    
    def evaluate(self, *_):
        raise NotImplementedError("Evaluate method is not implemented for TorchAlgorithm. Please implement it in the subclass.")
    

class TorchVAEAlgorithm(TorchNNAlgorithm, torch.nn.Module):
    """
    Algorithm class for Variational Autoencoders (VAEs) using PyTorch.
    Inherits from TorchNNAlgorithm to leverage its training and evaluation methods.
    """
    def __init__(self, algorithm: torch.nn.Module):
        torch.nn.Module.__init__(self)
        TorchNNAlgorithm.__init__(self, algorithm)
    
    def train(self, dataset: ROOTDataset,  **kwargs):
        return super().train(dataset, loss_func=VAELoss(), **kwargs)
    
    def training_step(self, optimizer, data, labels, criterion, epoch_loss, batch_count):
        data, labels = data.to(self.device), labels.to(self.device)
        optimizer.zero_grad()
        
        # Encode the input data
        x_hat, mean, log_var = self._algorithm(data)
 
        # Compute the loss
        loss = criterion(data, x_hat, mean, log_var, torch.nn.functional.mse_loss)
        
        loss.backward()
        optimizer.step()
        
        return loss