"""Check that the Keras 3 model is the same network as the original Keras 2 one.

The original DELIGHT_v1.h5 cannot be loaded by Keras 3 at all, so the
architecture was rebuilt in delight.delight.model and the trained weights were
copied across.  These tests are what justify calling the result the same model.
"""

import importlib.util
import os

import numpy as np
import pytest

import numpy_reference as ref
from conftest import ROOT, model_predict

# the legacy model: 5 inputs, a (None, 16) output, this many parameters
EXPECTED_PARAMS = 2297184
NLEVELS = 5
NVARIANTS = 8

# Both tolerances are absolute and in pixels on the 480x480 PanSTARRS cutout
# (0.25 arcsec per pixel), because the network predicts host offsets in those
# pixels. 1e-4 px is 25 microarcsec.
ORACLE_TOL = 1e-3   # px, Keras float32 against the float64 numpy reference
LEGACY_TOL = 1e-4   # px, Keras 3 against the original Keras 2 model


@pytest.fixture(scope="module")
def model_module():
    modelpy = os.path.join(ROOT, "delight", "delight", "model.py")
    spec = importlib.util.spec_from_file_location("delight_model_mod", modelpy)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_architecture_matches_legacy(delight_model):

    """shape and size of the rebuilt graph, against the original"""

    assert len(delight_model.inputs) == NLEVELS
    for tensor in delight_model.inputs:
        assert tuple(tensor.shape) == (None, 30, 30, 1)
    assert tuple(delight_model.output_shape) == (None, 2 * NVARIANTS)
    assert delight_model.count_params() == EXPECTED_PARAMS


def test_weight_shapes(delight_model):

    """the five layers that carry weights"""

    expected = {"conv2d_48": [(3, 3, 1, 52), (52,)],
                "conv2d_49": [(3, 3, 52, 57), (57,)],
                "conv2d_50": [(3, 3, 57, 41), (41,)],
                "dense_32": [(3280, 685), (685,)],
                "dense_33": [(685, 2), (2,)]}
    for name, shapes in expected.items():
        assert [w.shape for w in delight_model.get_layer(name).get_weights()] == shapes


def test_trunk_is_shared(delight_model):

    """the conv trunk is applied to all nlevels * nvariants images

    If the layers were not shared the parameter count would be far larger, so
    this is really a check that the graph reuses one trunk rather than 40.
    """

    for name in ("conv2d_48", "conv2d_49", "conv2d_50"):
        assert len(delight_model.get_layer(name)._inbound_nodes) == NLEVELS * NVARIANTS
    for name in ("dense_32", "dropout_16", "dense_33"):
        assert len(delight_model.get_layer(name)._inbound_nodes) == NVARIANTS


def test_rot90_matches_tensorflow_convention(model_module):

    """the Rot90 and FlipLeftRight layers must match the original ops

    The original graph used tf.image.rot90, which turns counter-clockwise and
    equals np.rot90 over the spatial axes, and tf.image.flip_left_right, which
    reverses the width axis.

    Non-square probes are deliberate: keras.ops.rot90(k=2) disagrees with
    np.rot90 for non-square inputs in keras 3.15, so Rot90 composes single
    quarter turns instead. DELIGHT's own images are square, but a layer that is
    only accidentally right is worth not shipping.
    """

    for shape in ((2, 3, 4, 1), (2, 4, 4, 1)):
        probe = np.arange(int(np.prod(shape)), dtype="float32").reshape(shape)
        for k in range(4):
            got = np.asarray(model_module.Rot90(k=k)(probe))
            assert np.array_equal(got, np.rot90(probe, k, axes=(1, 2))), \
                "Rot90(k=%i) wrong for shape %s" % (k, shape)

        flipped = np.asarray(model_module.FlipLeftRight()(probe))
        assert np.array_equal(flipped, probe[:, :, ::-1, :])


def test_matches_numpy_reference(delight_model, Xpr_golden, legacy_h5):

    """the rebuilt model against an independent float64 NumPy implementation

    The reference reads the weights straight from the legacy file and involves
    no Keras, so agreement pins down the variant ordering, the level ordering,
    the concatenation axes, the kernel orientation and the flatten order.
    """

    weights = ref.read_weights(legacy_h5)
    oracle = ref.predict(Xpr_golden, weights)
    got = model_predict(delight_model, Xpr_golden).astype("float64")

    assert got.shape == oracle.shape
    # ORACLE_TOL is in pixels, like everything the network emits. float32 Keras
    # against float64 NumPy, so the band is loose; a mis-ordered variant would
    # be O(1) px, four orders of magnitude away
    assert np.abs(got - oracle).max() < ORACLE_TOL

    # and per variant, so a failure names the one that is wrong
    for variant in range(NVARIANTS):
        columns = slice(2 * variant, 2 * variant + 2)
        assert np.abs(got[:, columns] - oracle[:, columns]).max() < ORACLE_TOL


def test_matches_legacy_keras2_output(delight_model, Xpr_golden, y_raw_legacy):

    """the rebuilt model against the original Keras 2 model's own output

    y_raw_legacy was produced by TensorFlow 2.15 loading the original
    DELIGHT_v1.h5 and running it on exactly this input, so this compares the
    two models directly with no preprocessing in between.
    """

    got = model_predict(delight_model, Xpr_golden).astype("float64")
    assert got.shape == y_raw_legacy.shape

    diff = np.abs(got - y_raw_legacy)
    # identical float32 weights and identical operations: the only slack is
    # accumulation order between TensorFlow versions
    assert diff.max() < LEGACY_TOL, "max abs diff %.3e px" % diff.max()


def test_weights_file_round_trips(delight_model, weights_file, model_module, Xpr_golden):

    """reloading the shipped weights file reproduces the model exactly"""

    reloaded = model_module.load_delight_model(weights_file)
    probe = Xpr_golden[:4]
    assert np.array_equal(model_predict(delight_model, probe),
                          model_predict(reloaded, probe))
