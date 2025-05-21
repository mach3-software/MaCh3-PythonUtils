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
        
        self._model.compile()
        self._model.to(self.device)
        


    def train_model(self):
        scaled_data = self.to_tensor(self.scale_data(self._training_data))
        scaled_labels = self.to_tensor(self.scale_labels(self._training_labels))

        loss = torch.nn.MSELoss()
        self._learning_rate = self._fit_settings.get("learning_rate", 1e-5)
        learning_rate_min = self._fit_settings.get("learning_rate_min", 1e-9)
        learning_rate_decay = self._fit_settings.get("learning_rate_decay", 0.1)
        
        num_epochs = self._fit_settings.get("num_epochs", 100)
        debug = self._fit_settings.get("debug", False)

        stop_on_plateau = self._fit_settings.get("stop_on_plateau", True)

        # Debug
        self._mean_weights_per_epoch = []*num_epochs  # Store mean weights per epoch
        self._mean_abs_weights_per_epoch = []*num_epochs

        self._loss_vals = np.zeros(num_epochs)


        for i in (pbar:=tqdm_notebook(range(num_epochs), desc="Training model", unit="epoch")):
            self.model_training_iter(scaled_data, scaled_labels, loss, self._learning_rate, debug, i)
            pbar.set_description(f"Epoch {i+1}/{num_epochs} - Loss: {self._loss_vals[i]:.5f}")

            # Early stopping
            if i < 10 or not stop_on_plateau:
                continue

            if self.early_stopping(loss, self._learning_rate, learning_rate_min, learning_rate_decay):
                break
            
    
    def early_stopping(self, loss, learning_rate, learning_rate_min, learning_rate_decay):
        """Early stopping function to stop training if loss does not improve

        :param loss: Loss function
        :type loss: torch.nn.Module
        :param learning_rate: Learning rate
        :type learning_rate: float
        :param learning_rate_min: Minimum learning rate
        :type learning_rate_min: float
        :param learning_rate_decay: Learning rate decay factor
        :type learning_rate_decay: float
        """
        
        if all(abs(self._loss_vals[-10:] - self._loss_vals[-1]) < 1e-5):
            if learning_rate > learning_rate_min:
                learning_rate *= learning_rate_decay

            else:
                print("Stopping training early")
                self._loss_vals = self._loss_vals[:i]
                return True
        
        return False

    def model_training_iter(self, scaled_data, scaled_labels, loss, learning_rate, debug, epoch):
        y_pred = self._model(scaled_data)
        loss_value = loss(y_pred, scaled_labels)
        self._model.zero_grad()
        loss_value.backward()


        with torch.no_grad():
            epoch_weights = []
            epoch_abs_weights = []

            for param in self._model.parameters():
                param -= learning_rate * param.grad
                if not debug:
                    continue
                
                # Store mean weights
                if param.requires_grad and param.dim() > 1:  # Only weights (not biases)
                    epoch_weights.append(param.data.mean().item())
                    epoch_abs_weights.append(param.data.abs().mean().item())

        self._loss_vals[epoch] = loss_value.item()

        if epoch_weights and debug:  # Only if weights were found
            self._mean_weights_per_epoch.append(np.mean(epoch_weights))
            self._mean_abs_weights_per_epoch.append(np.mean(epoch_abs_weights))
        elif debug:
            self._mean_weights_per_epoch.append(0.0)
            self._mean_abs_weights_per_epoch.append(0.0)

        
    def to_tensor(self, data):
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
        if isinstance(test_data, pd.DataFrame):
            test_data = torch.tensor(test_data.values.astype(np.float32), device=self.device)
        
        self._model.eval()
        with torch.no_grad():
            predictions = self._model(torch.tensor(test_data, device=self.device))
            
        return predictions.cpu().numpy().T[0]

    def print_model_summary(self):
        print("Model Summary:")
        print(f"Model: {self._model}")
        print(f"Device: {self.device}")
        
        # Plot loss
        plt.plot(self._loss_vals, color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()
        
        self.plot_weight_trends()
        

    def plot_weight_trends(self):
        """Plot mean weights and absolute weights over epochs."""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self._mean_weights_per_epoch, color="blue")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Weight")
        plt.title("Mean Weight per Epoch")
        
        plt.subplot(1, 2, 2)
        plt.plot(self._mean_abs_weights_per_epoch, color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Absolute Weight")
        plt.title("Mean Absolute Weight per Epoch")
        
        plt.tight_layout()
        plt.show()

    def save_model(self, path: str) -> None:
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path: str) -> None:
        """Load model weights"""
        self.model.load_state_dict(torch.load(path))
