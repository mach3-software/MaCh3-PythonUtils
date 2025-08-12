from MaCh3PythonUtils.machine_learning.file_ml_interface import FileMLInterface
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from rich import print
from matplotlib import pyplot as plt

class TorchInterface(FileMLInterface):
    __TORCH_LAYER_IMPLEMENTATIONS = {
        "linear": torch.nn.Linear,
        "relu": torch.nn.ReLU,
        "sigmoid": torch.nn.Sigmoid,
        "tanh": torch.nn.Tanh,
        "softmax": torch.nn.Softmax,
        "leaky_relu": torch.nn.LeakyReLU,
        "dropout": torch.nn.Dropout,
        "batch_norm": torch.nn.BatchNorm1d
    }
    
    _layers = []
    _training_settings = {}
    _loss_vals = []
    _learning_rates = []  # Track learning rate changes
    _mean_weights_per_epoch = []
    _mean_abs_weights_per_epoch = []
    
    # Set device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    
    def add_layer(self, layer_id: str, layer_args: dict):
        """Add new layer to PyTorch model

        :param layer_id: Layer type [linear/relu/sigmoid/tanh/softmax/leaky_relu/dropout/batchnorm]
        :type layer_id: str
        :param layer_args: kwargs for layer
        :type layer_args: dict
        :raises ValueError: Layer type not implemented in __TORCH_LAYER_IMPLEMENTATIONS yet
        """        
        
        if layer_id not in self.__TORCH_LAYER_IMPLEMENTATIONS.keys():
            raise ValueError(f"{layer_id} not implemented yet!")

        self._layers.append(self.__TORCH_LAYER_IMPLEMENTATIONS[layer_id.lower()](**layer_args))
        
    def build_model(self, **kwargs):
        """Build and compile PyTorch model

        :param kwargs: Model arguments as dictionary
        :type kwargs: dict
        :raises ValueError: Model not set up yet
        """
        self._model = torch.nn.Sequential(*self._layers)

        if self._model is None or not self._layers:
            raise ValueError("No model can be built! Please setup model and layers")

        self._model.to(self.device)
        self._fit_settings = kwargs
        
        # Initialize weights using Xavier/He initialization
        self._initialize_weights()
        
        # Don't call compile() - it's not a standard PyTorch method
        self._model.to(self.device)

    def _initialize_weights(self):
        """Initialize model weights for better training stability"""
        for layer in self._model:
            if isinstance(layer, torch.nn.Linear):
                # He initialization for ReLU networks, Xavier for others
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

    def train_model(self):
        print("Training Model")

        scaled_data = self.to_tensor(self.scale_data(self._training_data))
        scaled_labels = self.to_tensor(self._training_labels)

        # Setup loss function
        loss_fn = torch.nn.MSELoss()
        
        # Get training parameters
        self._learning_rate = self._fit_settings.get("learning_rate", 1e-3)  # Better default
        learning_rate_min = self._fit_settings.get("learning_rate_min", 1e-6)
        learning_rate_decay = self._fit_settings.get("learning_rate_decay", 0.5)
        weight_decay = self._fit_settings.get("weight_decay", 1e-4)  # L2 regularization
        
        optimizer_type = self._fit_settings.get("optimizer", "adam").lower()
        momentum = self._fit_settings.get("momentum", 0.9)
        
        num_epochs = self._fit_settings.get("num_epochs", 100)
        debug = self._fit_settings.get("debug", False)
        batch_size = self._fit_settings.get("batch_size", None)  # Full batch by default
        
        gradient_clip_norm = self._fit_settings.get("gradient_clip_norm", 1.0)
        stop_on_plateau = self._fit_settings.get("stop_on_plateau", True)
        patience = self._fit_settings.get("patience", 10)  # Epochs to wait before reducing LR

        # Setup optimizer
        self._optimizer = self._create_optimizer(optimizer_type, weight_decay, momentum)
        
        # Setup learning rate scheduler
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, 
            mode='min', 
            factor=learning_rate_decay, 
            patience=patience//2,
            min_lr=learning_rate_min,
        )

        # Initialize tracking arrays
        self._loss_vals = []
        self._learning_rates = []
        self._mean_weights_per_epoch = [] if debug else None
        self._mean_abs_weights_per_epoch = [] if debug else None

        # Create data loader if batch_size is specified
        if batch_size is not None:
            dataset = torch.utils.data.TensorDataset(scaled_data, scaled_labels)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            dataloader = [(scaled_data, scaled_labels)]

        best_loss = float('inf')
        patience_counter = 0

        for epoch in (pbar := tqdm_notebook(range(num_epochs), desc="Training model", unit="epoch")):
            epoch_loss = 0.0
            num_batches = 0
            
            self._model.train()
            
            for batch_data, batch_labels in dataloader:
                loss_value = self.model_training_iter(
                    batch_data, batch_labels, loss_fn, gradient_clip_norm, debug, epoch
                )
                epoch_loss += loss_value
                num_batches += 1
            
            # Average loss over batches
            avg_loss = epoch_loss / num_batches
            self._loss_vals.append(avg_loss)
            self._learning_rates.append(self._optimizer.param_groups[0]['lr'])
            
            # Update learning rate scheduler
            self._scheduler.step(avg_loss)
            
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f} - LR: {self._learning_rates[-1]:.2e}")

            # Early stopping logic
            if epoch >= patience and stop_on_plateau:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience and self._learning_rates[-1] <= learning_rate_min:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    def _create_optimizer(self, optimizer_type, weight_decay, momentum):
        """Create optimizer based on specified type"""
        if optimizer_type == "adam":
            return torch.optim.Adam(
                self._model.parameters(), 
                lr=self._learning_rate, 
                weight_decay=weight_decay
            )
        elif optimizer_type == "adamw":
            return torch.optim.AdamW(
                self._model.parameters(), 
                lr=self._learning_rate, 
                weight_decay=weight_decay
            )
        elif optimizer_type == "sgd":
            return torch.optim.SGD(
                self._model.parameters(), 
                lr=self._learning_rate, 
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_type == "rmsprop":
            return torch.optim.RMSprop(
                self._model.parameters(), 
                lr=self._learning_rate, 
                weight_decay=weight_decay
            )
        else:
            print(f"Unknown optimizer {optimizer_type}, using Adam")
            return torch.optim.Adam(
                self._model.parameters(), 
                lr=self._learning_rate, 
                weight_decay=weight_decay
            )

    def model_training_iter(self, batch_data, batch_labels, loss_fn, gradient_clip_norm, debug, epoch):
        """Single training iteration"""
        self._optimizer.zero_grad()
        
        y_pred = self._model(batch_data)
        loss_value = loss_fn(y_pred, batch_labels)
        
        loss_value.backward()
        
        # Gradient clipping for stability
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), gradient_clip_norm)
        
        self._optimizer.step()

        # Debug tracking
        if debug:
            with torch.no_grad():
                epoch_weights = []
                epoch_abs_weights = []
                
                for param in self._model.parameters():
                    if param.requires_grad and param.dim() > 1:  # Only weights (not biases)
                        epoch_weights.append(param.data.mean().item())
                        epoch_abs_weights.append(param.data.abs().mean().item())

                if epoch_weights:
                    self._mean_weights_per_epoch.append(np.mean(epoch_weights))
                    self._mean_abs_weights_per_epoch.append(np.mean(epoch_abs_weights))
                else:
                    self._mean_weights_per_epoch.append(0.0)
                    self._mean_abs_weights_per_epoch.append(0.0)

        return loss_value.item()
        
    def to_tensor(self, data):
        """Convert data to tensor"""
        if isinstance(data, pd.DataFrame):
            data = torch.tensor(data.values.astype(np.float32), device=self.device)
        elif isinstance(data, np.ndarray):
            data = torch.tensor(data.astype(np.float32), device=self.device)
        
        return data

    def model_predict(self, test_data):
        """Predict using the model

        :param test_data: Data to predict
        :type test_data: np.ndarray
        :return: Predictions
        :rtype: np.ndarray
        """
        # Convert to tensor and move to device
        if not isinstance(test_data, torch.Tensor):
            test_tensor = self.to_tensor(test_data)
        else:
            test_tensor = test_data

        self._model.eval()
        with torch.no_grad():
            predictions = self._model(test_tensor)
            
        return predictions.cpu().numpy().flatten()

    def print_model_summary(self):
        """Print model summary and plot training metrics"""
        print("Model Summary:")
        print(f"Model: {self._model}")
        print(f"Device: {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self._model.parameters())}")
        print(f"Trainable parameters: {sum(p.numel() for p in self._model.parameters() if p.requires_grad)}")
        
        # Plot training metrics
        self.plot_training_metrics()
        
        if self._mean_weights_per_epoch is not None:
            self.plot_weight_trends()

    def plot_training_metrics(self):
        """Plot loss and learning rate over training"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self._loss_vals, color="orange", linewidth=2)
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax2.plot(self._learning_rates, color="green", linewidth=2)
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_weight_trends(self):
        """Plot mean weights and absolute weights over epochs."""
        if self._mean_weights_per_epoch is None:
            print("No weight tracking data available. Set debug=True during training.")
            return
            
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self._mean_weights_per_epoch, color="blue", linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("Mean Weight")
        plt.title("Mean Weight per Epoch")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self._mean_abs_weights_per_epoch, color="red", linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("Mean Absolute Weight")
        plt.title("Mean Absolute Weight per Epoch")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def save_model(self, path: str) -> None:
        """Save model weights and optimizer state"""
        torch.save({
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'loss_history': self._loss_vals,
            'lr_history': self._learning_rates
        }, path)
        
    def load_model(self, path: str) -> None:
        """Load model weights and optimizer state"""
        checkpoint = torch.load(path, map_location=self.device)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        
        if hasattr(self, '_optimizer') and 'optimizer_state_dict' in checkpoint:
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'loss_history' in checkpoint:
            self._loss_vals = checkpoint['loss_history']
        if 'lr_history' in checkpoint:
            self._learning_rates = checkpoint['lr_history']