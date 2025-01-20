from MaCh3PythonUtils.machine_learning.tensorflow.tf_manual_interface import (
    TfManualLayeredInterface,
)
import tensorflow.keras as tfk


class TfResidualModel(TfManualLayeredInterface):
    def build_model(self, **kwargs):
        input_shape = self.training_data.shape[
            1:
        ]  # Assuming shape is (batch_size, features)
        network_input = tfk.layers.Input(shape=input_shape)

        # Initial layer
        x = self._layers[0](network_input)

        # Apply each layer in self._layers, adding residual connections every other layer
        for i, layer in enumerate(self._layers[1:], start=1):
            # Apply layer
            x_new = layer(x)

            # Add residual connection every other layer if shapes match
            if i % 2 == 0 and x.shape == x_new.shape:
                x = tfk.layers.add([x, x_new])
            else:
                x = x_new  # Update x with current layer output

        # Define and compile the model
        self._model = tfk.Model(inputs=network_input, outputs=x)
        optimizer = tfk.optimizers.AdamW(
            learning_rate=kwargs.get("learning_rate", 1e-5),
            weight_decay=1e-4,
            clipnorm=1.0,
        )

        _ = kwargs.pop("learning_rate", None)

        self._model.compile(optimizer=optimizer, **kwargs)
