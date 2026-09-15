"""Keras 3 implementation of the DELIGHT convolutional neural network.

The original model was trained with Keras 2.7 and distributed as
``DELIGHT_v1.h5``.  That file stores the rotation/flip/concatenate operations
as ``TFOpLambda`` layers, a Keras 2 serialization wrapper with no Keras 3
equivalent, so it can no longer be deserialized.

The architecture is therefore defined here in code and the trained weights are
loaded separately from ``DELIGHT_v1.weights.h5``.  Keeping the architecture in
code means no layer configuration is ever deserialized, so a future Keras
release cannot break loading the way Keras 3 broke the original file.

The layer names below are the ones used in the original model, so weights map
by name.
"""

import keras
from keras import layers

NLEVELS = 5
IMSIZE = 30

# The eight dihedral variants each input is passed through, in the order the
# original graph applied them (traced through its flatten_640..679 layers).
# Each entry is (flip_left_right_first, number_of_counter-clockwise_rot90).
#
# This order must match delight.Delight.derotate(), which undoes each transform
# by index.  Getting it wrong corrupts predictions silently.
VARIANTS = (
    (False, 0),
    (False, 1),
    (False, 2),
    (False, 3),
    (True, 0),
    (True, 1),
    (True, 2),
    (True, 3),
)


@keras.saving.register_keras_serializable(package="delight")
class Rot90(layers.Layer):

    """rotate an image batch counter-clockwise by 90 degrees k times"""

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def call(self, inputs):
        # Applied one quarter turn at a time, the way the original graph chained
        # its tf.image.rot90 layers. keras.ops.rot90(k=2) is also wrong for
        # non-square images in keras 3.15, and composing single turns avoids it.
        x = inputs
        for _ in range(self.k % 4):
            x = keras.ops.rot90(x, k=1, axes=(1, 2))
        return x

    def compute_output_shape(self, input_shape):
        batch, height, width, channels = input_shape
        if self.k % 2:
            return (batch, width, height, channels)
        return (batch, height, width, channels)

    def get_config(self):
        config = super().get_config()
        config["k"] = self.k
        return config


@keras.saving.register_keras_serializable(package="delight")
class FlipLeftRight(layers.Layer):

    """mirror an image batch along its width axis"""

    def call(self, inputs):
        return keras.ops.flip(inputs, axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape


def build_delight_model(nlevels=NLEVELS, imsize=IMSIZE):

    """build the DELIGHT architecture with randomly initialised weights

    Parameters
    ----------
    nlevels : int
       number of multi-resolution levels, one input per level
    imsize : int
       side of each multi-resolution image in pixels
    """

    inputs = [layers.Input(shape=(imsize, imsize, 1), name="input_%i" % (81 + i))
              for i in range(nlevels)]

    # trunk, shared across all nlevels * 8 transformed images
    conv1 = layers.Conv2D(52, (3, 3), activation="relu", name="conv2d_48")
    pool1 = layers.MaxPooling2D((2, 2), name="max_pooling2d_32")
    conv2 = layers.Conv2D(57, (3, 3), activation="relu", name="conv2d_49")
    pool2 = layers.MaxPooling2D((2, 2), name="max_pooling2d_33")
    conv3 = layers.Conv2D(41, (3, 3), activation="relu", name="conv2d_50")
    flatten = layers.Flatten(name="flatten")

    # head, shared across the 8 variants
    dense1 = layers.Dense(685, activation="tanh",
                          kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                          name="dense_32")
    dropout = layers.Dropout(0.062505, name="dropout_16")
    dense2 = layers.Dense(2, name="dense_33")

    variant_outputs = []
    for flip, k in VARIANTS:
        features = []
        for level in range(nlevels):
            x = inputs[level]
            if flip:
                x = FlipLeftRight()(x)
            if k:
                x = Rot90(k=k)(x)
            features.append(flatten(conv3(pool2(conv2(pool1(conv1(x)))))))
        # concatenate the nlevels resolution levels, then predict this variant
        x = layers.Concatenate(axis=1)(features)
        variant_outputs.append(dense2(dropout(dense1(x))))

    # (batch, 2 * 8), unpacked by Delight.derotate()
    outputs = layers.Concatenate(axis=1)(variant_outputs)

    return keras.Model(inputs=inputs, outputs=outputs, name="DELIGHT")


def load_delight_model(weightsfile, nlevels=NLEVELS, imsize=IMSIZE):

    """build the architecture and load trained weights into it

    Parameters
    ----------
    weightsfile : string
       path to a DELIGHT .weights.h5 file
    """

    model = build_delight_model(nlevels=nlevels, imsize=imsize)
    model.load_weights(weightsfile)
    return model
