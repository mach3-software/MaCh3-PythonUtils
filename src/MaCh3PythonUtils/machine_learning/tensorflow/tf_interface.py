# Let's make a tensor flow interface!
from MaCh3PythonUtils.machine_learning.file_ml_interface import FileMLInterface
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow.keras as tfk
import tensorflow_probability as tfp
from typing import Iterable


class TfInterface(FileMLInterface):
    __TF_LAYER_IMPLEMENTATIONS = {
        "dense": tfk.layers.Dense,
        "dropout": tfk.layers.Dropout,
        "batchnorm": tfk.layers.BatchNormalization,
    }

    _layers = []
    _training_settings = {}

    def add_layer(self, layer_id: str, layer_args: dict):
        """Add new layer to TF model

        :param layer_id: Layer type [dense/dropout]
        :type layer_id: str
        :param layer_args: kwargs for layer
        :type layer_args: dict
        :raises ValueError: Layer type not implemented in __TF_LAYER_IMPLEMENTATIONS yet
        """
        if layer_id not in self.__TF_LAYER_IMPLEMENTATIONS.keys():
            raise ValueError(f"{layer_id} not implemented yet!")

        if layer_args.get("kernel_regularizer"):
            # Hacky, swaps string value of regularliser for proper one
            layer_args["kernel_regularizer"] = tfk.regularizers.L2(0.2)

        self._layers.append(
            self.__TF_LAYER_IMPLEMENTATIONS[layer_id.lower()](**layer_args)
        )

    def build_model(self, **kwargs):
        return None

    def set_training_settings(self, kwargs):
        """Set training settings, needs to be done early for...reasons

        :param kwargs: training kwargs
        :type kwargs: kwargs
        """
        self._training_settings = kwargs

    def train_model(self):
        """train model"""
        scaled_data = self.scale_data(self._training_data)

        lr_schedule = tfk.callbacks.ReduceLROnPlateau(
            monitor="loss", patience=5, factor=0.5, min_lr=1e-6, verbose=1
        )

        self._model.fit(
            scaled_data,
            self._training_labels,
            **self._training_settings,
            callbacks=[lr_schedule],
        )
        print(f"Using loss function: {self._model.loss}")

    def save_model(self, output_file: str):
        """Save model to file

        :param output_file: Output file to save model to
            :type output_file: str
        """
        if not ".keras" in output_file:
            output_file += ".keras"

        self._model.save(
            output_file,
        )

    def load_model(self, input_file: str):
        """Load model from file

        :param input_file: Name offile to load model from
        :type input_file: str
        """

        print(f"Loading model from {input_file}")
        self._model = tf.keras.models.load_model(input_file)

    def model_predict(self, test_data: pd.DataFrame):
        """Get model prediction

        :param test_data: data to run model over for prediction
        :type test_data: pd.DataFrame
        :return: model predction
        :rtype: NDArray
        """
        # Hacky but means it's consistent with sci-kit interface
        scaled_data = self.scale_data(test_data)

        if self._model is None:
            return np.zeros(len(test_data))

        return self._model.predict(scaled_data, verbose=False).T[0]

    def model_predict_no_scale(self, test_data):
        # Same as above but specifically for TF, optimised to avoid if statement...
        return self._model(test_data, training=False)

    def evaluate_model(
        self, predicted_values: Iterable, true_values: Iterable, outfile: str = ""
    ):

        # CODE TO DO TF SPECIFIC PLOTS GOES HERE

        return super().evaluate_model(predicted_values, true_values, outfile)
