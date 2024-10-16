import normflows as nf
import torch
from tqdm import tqdm
import pandas as pd
from numpy.typing import NDArray
from typing import List, Any, Tuple
import numpy as np

# Series of algorithms that use the normalising flow package

class __NormalisingFlowAlgorithmBase:
    # Assuming all normalising flow algorithms use the same structure
    
    _loss_function = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, n_dim: int, n_iter: int=1000, verbose: bool=False) -> None:
        self._n_iter = n_iter
        self._verbose = verbose
        self._base = nf.distributions.base.DiagGaussian(tuple([0]*n_dim))
        self._flows = []
        self._model: nf.NormalizingFlow | None = None
    
    @property
    def model(self)->Any:
        return self._model
    
    def build_model(self):
        self._model = nf.NormalizingFlow(self._base, self._flows)

    
    def fit(self, training_data: NDArray, training_labels: pd.DataFrame):
        """Fit model

        :param training_data: _description_
        :type training_data: pd.DataFrame
        :param training_labels: _description_
        :type training_labels: pd.DataFrame
        :raises Exception: _description_
        """ 
        if self._model is None:
            raise Exception("Model hasn't been set")
        
        print("Fitting Model")
        
        self._model = self._model.to(self.device)
        self._model = self._model.double()

        x = torch.tensor(training_data, dtype=torch.double).to(self.device)
        true_log_likelihood = torch.tensor(training_labels.values, dtype=torch.double).squeeze().to(self.device)


        self.build_model

        print("Initialising Optimizer")
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)

        for epoch in tqdm(range(self._n_iter)):
            optimizer.zero_grad()

            # Get the log probability from the model
            predicted_log_prob = self._model.log_prob(x)

            # Calculate loss as the negative log likelihood
            loss = torch.mean((predicted_log_prob - true_log_likelihood) ** 2)

            # Perform backpropagation and optimization step
            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

        print("Training complete.")

    def predict(self, test_data: NDArray):
        if self._model is None: 
            raise Exception("Model hasn't been set")

        prediction = self._model.log_prob(torch.tensor(test_data, dtype=torch.double).to(self.device))
        
        # Convert to correct format!
        return prediction.detach().numpy()

class RealNVPModel(__NormalisingFlowAlgorithmBase):
    # Basic implementation from https://pypi.org/project/normflows/
    def __init__(self, n_dim: int, n_layers: int=1, layer_structure: List[int]=[64, 1], n_iter: int=1000, verbose: bool=False):

        super().__init__(n_dim=n_dim, n_iter=n_iter, verbose=verbose)

        # Taken from https://github.com/VincentStimper/normalizing-flows/blob/master/examples/real_nvp.ipynb

        print(f"Number of dimensions: {n_dim}")

        b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(n_dim)])
        flows = []
        for i in range(n_layers):
            s = nf.nets.MLP(layer_structure, init_zeros=True)
            t = nf.nets.MLP(layer_structure, init_zeros=True)
            if i % 2 == 0:
                flows += [nf.flows.MaskedAffineFlow(b, t, s)]
            else:
                flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
            flows += [nf.flows.ActNorm(n_dim)]

        
        self.build_model()
        
class Glow(__NormalisingFlowAlgorithmBase):
    def __init__(self, n_dim: Tuple[int], n_iter: int=1000, top_layers: int=3, lower_layers: int=10, channels=3, hidden_channels=256, split_mode='channel', scale=True, num_classes=10, verbose: bool=False):
        super().__init__(n_dim[0], n_iter, verbose)
        
        q0 = []
        merges = []
        flows = []
        
        for i in range(top_layers):
            flows_ = []
            for j in range(lower_layers):
                flows_ += [nf.flows.GlowBlock(channels*2 **(top_layers+1-i), hidden_channels, split_mode=split_mode, scale=scale)]
                
                flows_ += [nf.flows.Squeeze()]
                flows += flows_
                if i>0:
                    merges += [nf.flows.Merge()]
                    latent_shape = ()