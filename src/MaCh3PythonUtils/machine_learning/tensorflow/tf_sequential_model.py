from MaCh3PythonUtils.machine_learning.tensorflow.tf_manual_interface import TfManualLayeredInterface

import tensorflow.keras as tfk

class TfSequentialModel(TfManualLayeredInterface):
    
    def build_model(self, model_args: dict):
        """Build and compile TF model

        :param model_args: Model arguments as dictionary
        :type model_args: dict
        :raises ValueError: Model not set up yet
        """
        self._model = tfk.Sequential()

        if self._model is None or not self._layers:
            raise ValueError("No model can be built! Please setup model and layers")

        for layer in self._layers:
            self._model.add(layer)

        self._model.build()
        optimizer = tfk.optimizers.AdamW(learning_rate=model_args.get("learning_rate", 1e-5),
                        weight_decay=1e-4, clipnorm=1.0)

        model_args.pop("learning_rate", None)


        self._model.compile(**model_args, optimizer=optimizer)