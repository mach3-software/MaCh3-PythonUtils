from MaCh3PythonUtils.file_handling.chain_handler import ChainHandler

from abc import ABC, abstractmethod
from typing import Any, Tuple, Iterable
from sklearn.model_selection import train_test_split
import pandas as pd
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pickle

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class FileMLInterface(ABC):
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
        """General Interface for all ML models

        :param chain: ChainHandler instance
        :type chain: ChainHandler
        :param prediction_variable: "Label" used for prediction
        :type prediction_variable: str
        :raises ValueError: Checks to see if label exists in tree
        """        
        self._chain = chain
        
        self._prediction_variable = prediction_variable
        
        if prediction_variable not in self._chain.ttree_array.columns:
            raise ValueError(f"Cannot find {prediction_variable} in input tree")
        
        self._model = None
        
        self._training_data=None
        self._training_labels=None
        self._test_data=None
        self._test_labels=None
        self._scalar = StandardScaler()
            
    def __separate_dataframe(self)->Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data frame into feature + label objects

        :return: features, labels
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """        
        # Separates dataframe into features + labels
        features = self._chain.ttree_array.copy()
        labels   = pd.DataFrame(features.pop(self._prediction_variable) )
        
        return features, labels
    
    def set_training_test_set(self, test_size: float):
        """Splits data/labels into training and testing tests

        :param test_size: Proportion of data used for testing
        :type test_size: float
        """        
        # Splits in traing + test_spit
        features, labels = self.__separate_dataframe()
        self._training_data, self._test_data, self._training_labels, self._test_labels =  train_test_split(features, labels, test_size=test_size)
        self._training_data = self._scalar.fit_transform(self._training_data)


    @property
    def model(self)->Any:
        """Model used

        :return: Returns ML model being used
        :rtype: Any
        """        
        # Returns model being used
        return self._model    
    
    @property
    def training_data(self)->pd.DataFrame:
        """Gets training data

        :return: Training data set
        :rtype: pd.DataFrame
        """        
        return self._training_data
    
    def add_model(self, ml_model: Any)->None:
        """Add model to data set

        :param ml_model: Sets model to be ml_model
        :type ml_model: Any
        """        
        # Add ML model into your interface
        self._model = ml_model
    
    @abstractmethod
    def train_model(self):
        """Abstract method, should be overwritten with model training
        """        
        # Train Model method
        pass
    
    @abstractmethod
    def model_predict(self, testing_data: pd.DataFrame)->Iterable:
        """Abstract method, should return model prediction

        :param testing_data: Data to test model on 
        :type testing_data: pd.DataFrame
        """        
        pass
        
    def save_model(self, output_file: str):
        """Save model to pickle

        :param output_file: Pickle file to save to
        :type output_file: str
        """        
        print(f"Saving to {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(self._model, f)
        
    def load_model(self, input_file: str):
        """Unpickle model

        :param input_file: Pickled Model
        :type input_file: str
        """        
        print(f"Attempting to load file from {input_file}")
        with open(input_file, 'r') as f:
            self._model = pickle.load(f)
        
    def test_model(self):
        """Test model

        :raises ValueError: No model set
        :raises ValueError: No test data set 
        """        
        if self._model is None:
            raise ValueError("No Model has been set!")

        if self._test_data is None or self._test_labels is None:
            raise ValueError("No test data set")

        prediction = self.model_predict(self._test_data)
        test_as_numpy = self._test_labels.to_numpy().T[0]
        
        self.evaluate_model(prediction, test_as_numpy)
            
    
    def evaluate_model(self, predicted_values: Iterable, true_values: Iterable, outfile: str=""):
        """Evalulates model

        :param predicted_values: Label values predicted by model
        :type predicted_values: Iterable
        :param true_values: Actual label values
        :type true_values: Iterable
        :param outfile: File to output plots to, defaults to ""
        :type outfile: str, optional
        """        
        print(predicted_values)
        print(true_values)
        
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
        plt.close()
        
        # Gonna draw a hist
        difs = true_values-predicted_values
        print(f"mean: {np.mean(difs)}, std dev: {np.std(difs)}")
        plt.hist(difs, bins=100, density=True, range=(np.std(difs)*-5, np.std(difs)*5))
        plt.xlabel("True - Pred")
        plt.savefig(f"diffs_5sigma_range_{outfile}")