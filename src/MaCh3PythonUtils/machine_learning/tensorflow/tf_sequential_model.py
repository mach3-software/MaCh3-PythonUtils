from MaCh3PythonUtils.machine_learning.tensorflow.tf_manual_interface import TfManualLayeredInterface

import tensorflow.keras as tfk

class TfSequentialModel(TfManualLayeredInterface):
    
    def build_model(self, **kwargs: dict):
        """Build and compile TF model

        :param kwargs: Model arguments as dictionary
        :type kwargs: dict
        :raises ValueError: Model not set up yet
        """
        self._model = tfk.Sequential()

        if self._model is None or not self._layers:
            raise ValueError("No model can be built! Please setup model and layers")

        # Add input layer
        self._model.add(tfk.layers.InputLayer(input_shape=(self._chain.ndim-1,)))

        for layer in self._layers:
            self._model.add(layer)

        self._model.build()
        optimizer = tfk.optimizers.Adam(learning_rate=kwargs.get("learning_rate", 1e-5), clipnorm=10.0)

        kwargs.pop("learning_rate", None)


        self._model.compile(**kwargs, optimizer=optimizer)
        