from abc import ABC, abstractmethod
import logging
from typing import Generic, TypeVar


import torch
from torch.utils.data import random_split

from mach3pythonutils.file_io.root_dataset import ROOTDataset

T = TypeVar('T')
class MLAlgorithm(ABC, Generic[T]):
    def __init__(self, algorithm: T):
        """
        Initialize the MLAlgorithm with a model.
        :param model: The machine learning model to use.
        """
        self._algorithm = algorithm
    
    @property
    def algorithm(self) -> T:
        return self._algorithm
    
    @abstractmethod
    def train(self, dataset: ROOTDataset, **kwargs):
        '''
        Define the training loop for the algorithm.
        '''
        pass

    @abstractmethod
    def evaluate(self, dataset: ROOTDataset, **kwargs):
        '''
        Evaluate ability of model to.
        '''
        pass

    @abstractmethod
    def predict(self, dataset: ROOTDataset, **kwargs):
        '''
        Get algorithm to predict labels for the given dataset.
        '''
        pass    
    
class MLCallBack(ABC):
    """
    Abstract base class for machine learning callbacks.
    Callbacks can be used to monitor training and validation processes.
    """
    def __init__(self, model: MLAlgorithm):
        """
        Initialize the MLCallBack with a machine learning model.
        
        :param model: The machine learning model to use.
        :type model: MLAlgorithm
        """
        self.model = model
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: dict):
        """
        Called at the end of each epoch.
        :param epoch: The current epoch number.
        :param logs: A dictionary containing metrics and other information.
        """
        pass
    
    @abstractmethod
    def on_train_end(self, logs: dict):
        """
        Called at the end of training.
        :param logs: A dictionary containing metrics and other information.
        """
        pass


class MLInterface():
    def __init__(self, dataset: ROOTDataset, model: MLAlgorithm, training_set_size: float = 0.8, generator_seed: int = 42):
        """
        Initialize the MLInterface with a dataset and a model.
        :param dataset: The dataset to use for training and validation.
        :param model: The machine learning model to use.
        :param training_set_size: The proportion of the dataset to use for training.
        :param generator_seed: Seed for random number generation to ensure reproducibility.
        """

        if training_set_size <= 0 or training_set_size >= 1:
            logging.critical("training_set_size must be between 0 and 1 (exclusive)", exc_info=True)
            raise ValueError("training_set_size must be between 0 and 1 (exclusive)")

        self._training_set, self._validation_set = random_split(
            dataset,
            [training_set_size, 1 - training_set_size],
            generator=torch.Generator().manual_seed(generator_seed)
        )

        self._model = model
   
    @property
    def model(self) -> MLAlgorithm:
        return self._model
    
    @property
    def training_set(self) -> ROOTDataset:
        return self._training_set
    
    @property
    def validation_set(self) -> ROOTDataset:
        return self._validation_set