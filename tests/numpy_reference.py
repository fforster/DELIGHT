"""An independent, pure NumPy implementation of the DELIGHT forward pass.

This exists to check the Keras 3 model without trusting Keras at all.  It reads
the trained weights straight out of the legacy Keras 2 ``DELIGHT_v1.h5`` and
reimplements the network in float64, so agreement between the two pins down
the variant ordering, the level ordering, the concatenation axes, the kernel
orientation and the flatten order.

Details that matter, and are easy to get wrong:

* Keras ``Conv2D`` is a cross-correlation, not a convolution: the kernel is
  *not* flipped.
* Data layout is channels_last and padding is ``valid`` throughout, so the
  spatial sizes run 30 -> 28 -> 14 -> 12 -> 6 -> 4.
* ``Flatten`` is C-order over (height, width, channels).
* The dihedral variants apply the left-right flip *before* the rotations, and
  ``tf.image.rot90`` rotates counter-clockwise, matching ``np.rot90``.
* ``Dropout`` is the identity at inference time.
"""

import h5py
import numpy as np

LAYERS = ("conv2d_48", "conv2d_49", "conv2d_50", "dense_32", "dense_33")

# (flip_left_right first, number of counter-clockwise rot90), in the order the
# original graph applied them
VARIANTS = (
    (False, 0), (False, 1), (False, 2), (False, 3),
    (True, 0), (True, 1), (True, 2), (True, 3),
)


def read_weights(h5file):

    """read the trained weights out of the legacy Keras 2 model file"""

    weights = {}
    with h5py.File(h5file, "r") as f:
        group = f["model_weights"]
        for name in LAYERS:
            weights[name] = (
                np.asarray(group["%s/%s/kernel:0" % (name, name)], dtype="float64"),
                np.asarray(group["%s/%s/bias:0" % (name, name)], dtype="float64"))
    return weights


def conv2d_relu(x, kernel, bias):

    """valid cross-correlation over a channels_last batch, then relu

    Parameters
    ----------
    x : numpy array
       input of shape (batch, height, width, channels)
    kernel : numpy array
       kernel of shape (kh, kw, in_channels, out_channels)
    bias : numpy array
       bias of shape (out_channels,)
    """

    batch, height, width, channels = x.shape
    kh, kw, kin, kout = kernel.shape
    if kin != channels:
        raise ValueError("kernel expects %i channels, got %i" % (kin, channels))
    oh, ow = height - kh + 1, width - kw + 1

    # gather every (kh, kw, channels) patch, then one matrix multiply
    patches = np.empty((batch, oh, ow, kh, kw, channels), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            patches[:, :, :, i, j, :] = x[:, i:i + oh, j:j + ow, :]

    out = patches.reshape(batch, oh, ow, kh * kw * channels) @ kernel.reshape(-1, kout)
    return np.maximum(out + bias, 0.0)


def maxpool2x2(x):

    """2x2 max pooling with stride 2, valid padding, channels_last"""

    batch, height, width, channels = x.shape
    height, width = height - height % 2, width - width % 2
    x = x[:, :height, :width, :]
    return x.reshape(batch, height // 2, 2, width // 2, 2, channels).max(axis=(2, 4))


def trunk(x, weights):

    """the shared convolutional trunk, returning flattened features"""

    x = conv2d_relu(x, *weights["conv2d_48"])
    x = maxpool2x2(x)
    x = conv2d_relu(x, *weights["conv2d_49"])
    x = maxpool2x2(x)
    x = conv2d_relu(x, *weights["conv2d_50"])
    return x.reshape(x.shape[0], -1)


def transform(x, flip, k):

    """apply one dihedral variant to a channels_last batch"""

    if flip:
        x = x[:, :, ::-1, :]
    if k:
        x = np.rot90(x, k, axes=(1, 2))
    return np.ascontiguousarray(x)


def predict(Xpr, weights):

    """run the whole network

    Parameters
    ----------
    Xpr : numpy array
       preprocessed input of shape (batch, 30, 30, nlevels)
    weights : dict
       output of read_weights

    Returns
    -------
    numpy array of shape (batch, 2 * 8), matching the Keras model's raw output
    """

    Xpr = np.asarray(Xpr, dtype="float64")
    nlevels = Xpr.shape[3]

    kernel32, bias32 = weights["dense_32"]
    kernel33, bias33 = weights["dense_33"]

    outputs = []
    for flip, k in VARIANTS:
        features = [trunk(transform(Xpr[:, :, :, level][..., np.newaxis], flip, k), weights)
                    for level in range(nlevels)]
        x = np.concatenate(features, axis=1)
        x = np.tanh(x @ kernel32 + bias32)          # dropout is the identity here
        outputs.append(x @ kernel33 + bias33)

    return np.concatenate(outputs, axis=1)
