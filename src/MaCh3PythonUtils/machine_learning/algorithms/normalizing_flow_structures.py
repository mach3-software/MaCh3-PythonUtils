import normflows as nf
from typing import List, Any
import pandas as pd
import torch
from tqdm import tqdm

# Series of algorithms that use the normalising flow package

class __NormalisingFlowAlgorithmBase:
    # Assuming all normalising flow algorithms use the same structure
    
    _loss_function = []
    
    def __init__(self, n_dim: int, n_iter: int=1000, verbose: bool=False) -> None:
        self._n_iter = n_iter
        self._verbose = verbose
        self._base = nf.distributions.base.DiagGaussian(n_dim)
        self._flows = []
        self._model: nf.NormalizingFlow | None = None
    
    @property
    def model(self)->Any:
        return self._model
    
    def build_model(self):
        self._model = nf.NormalizingFlow(self._base, self._flows)

    
    def fit(self, training_data: pd.DataFrame, training_labels: pd.DataFrame):
        if self._model is None:
            raise Exception("Model hasn't been set")
        
        x = torch.tensor(training_data.values, dtype=torch.float32)
        true_log_likelihood = torch.tensor(training_labels.values, dtype=torch.float32).squeeze()

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

    def predict(self, test_data: pd.DataFrame):
        if self._model is None:
            raise Exception("Model hasn't been set")

        prediction = self._model.log_prob(torch.tensor(test_data.values, dtype=torch.float32))
        
        # Convert to correct format!
        return prediction.detach().numpy().tolist()

class RealNVPModel(__NormalisingFlowAlgorithmBase):
    # Basic implementation from https://pypi.org/project/normflows/
    def __init__(self, n_dim: int, n_layers: int=1, layer_structure: List[int]=[64, 1], n_iter: int=1000, verbose: bool=False):

        super().__init__(n_dim=n_dim, n_iter=n_iter, verbose=verbose)

        for _ in range(n_layers):
            param_map = nf.nets.MLP(layer_structure, init_zeros=True)
            self._flows.append(nf.flows.AffineCouplingBlock(param_map))
            self._flows.append(nf.flows.Permute(n_dim, mode='swap'))
        
        self.build_model()