from abc import ABC, abstractmethod
from file_handling.chain_handler import ChainHandler
from typing import Any, Tuple
from sklearn.model_selection import train_test_split
import pandas as pd
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap

from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

import pickle

class FmlInterface(ABC):
    """Abstract interface with file handler which should be used for ML models
    """
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.3, '#2a788e'),
        (0.4, '#21a784'),
        (0.7, '#78d151'),
        (1, '#fde624'),
    ], N=256)
    
    def __init__(self, chain: ChainHandler, prediction_variable: str) -> None:
        self._chain = chain
        
        self._prediction_variable = prediction_variable
        
        if prediction_variable not in self._chain.ttree_array.columns:
            raise ValueError(f"Cannot find {prediction_variable} in input tree")
        
        self._model = None
        
        self._training_data=None
        self._training_labels=None
        self._test_data=None
        self._test_labels=None
            
    def __separate_dataframe(self)->Tuple[pd.DataFrame, pd.DataFrame]:
        # Separates dataframe into features + labels
        features = self._chain.ttree_array.copy()
        labels   = pd.DataFrame(features.pop(self._prediction_variable) )
        
        return features, labels
    
    def set_training_test_set(self, test_size: float):
        # Splits in traing + test_spit
        features, labels = self.__separate_dataframe()
        self._training_data, self._test_data, self._training_labels, self._test_labels =  train_test_split(features, labels, test_size=test_size)

    @property
    def model(self)->Any:
        # Returns model being used
        return self._model    
    
    def add_model(self, ml_model: Any)->None:
        # Add ML model into your interface
        self._model = ml_model
    
    @abstractmethod
    def train_model(self):
        # Train Model method
        pass
    
    
    @abstractmethod
    def model_predict(self, testing_data):
        # Run model prediction
        pass
    
    @abstractmethod
    def test_model(self):
        # Test Model
        pass
    
    def save_model(self, output_file: str):
        print(f"Saving to {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(self._model, f)
        
    def load_model(self, input_file: str):
        print(f"Attempting to load file from {input_file}")
        with open(input_file, 'r') as f:
            self._model = pickle.load(f)
    
    def evaluate_model(self, predicted_values, true_values, outfile: str=""):
        print(f"Mean Absolute Error : {metrics.mean_absolute_error(predicted_values,true_values)}")
        
        lobf = np.poly1d(np.polyfit(predicted_values, true_values, 1))
        
        print(f"Line of best fit : y={lobf.c[0]}x + {lobf.c[1]}")
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='scatter_density')

        density = ax.scatter_density(predicted_values, true_values, cmap=self.white_viridis)
        fig.colorbar(density, label="number of points per pixel")
        
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        ax.plot(lims, lobf(lims), "w-", label=f"Best fit: true={lobf.c[0]}pred + {lobf.c[1]}", linestyle="dashed", linewidth=0.3)

        ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0, label="true=predicted", linestyle="dashed", linewidth=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        
        ax.set_xlabel("Predicted Log likelihood")
        ax.set_ylabel("True Log Likelihood")
        
        fig.legend()
        if outfile=="": outfile = "evaluated_model_qq.pdf"
        
        print(f"Saving QQ to {outfile}")
            
        fig.savefig(outfile)