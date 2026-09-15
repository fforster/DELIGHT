"""Shared fixtures for the DELIGHT test suite.

The suite runs offline against the golden files in tests/golden, which were
produced by the original TensorFlow 2.15 / Keras 2 version of the library.
"""

import os
import sys

# keep the comparisons reproducible across machines and TensorFlow builds
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLDEN = os.path.join(HERE, "golden")

sys.path.insert(0, HERE)


@pytest.fixture(scope="session")
def golden_dir():
    return GOLDEN


@pytest.fixture(scope="session")
def legacy_h5():

    """the original Keras 2 model file, the source of the trained weights"""

    path = os.path.join(ROOT, "delight", "delight", "DELIGHT_v1.h5")
    if not os.path.exists(path):
        pytest.skip("legacy model file not available")
    return path


@pytest.fixture(scope="session")
def weights_file():

    """the converted Keras 3 weights the library ships"""

    path = os.path.join(ROOT, "delight", "delight", "DELIGHT_v1.weights.h5")
    if not os.path.exists(path):
        pytest.skip("converted weights not found, run tools/convert_model.py")
    return path


@pytest.fixture(scope="session")
def Xpr_golden():

    """the preprocessed network input from the reference run"""

    return np.load(os.path.join(GOLDEN, "Xpr_golden.npz"))["Xpr"]


@pytest.fixture(scope="session")
def y_raw_legacy():

    """raw network output of the original Keras 2 model on Xpr_golden"""

    return np.load(os.path.join(GOLDEN, "y_raw_legacy.npz"))["y_raw"]


@pytest.fixture(scope="session")
def baseline():

    """the reference end-to-end run of the original version"""

    return np.load(os.path.join(GOLDEN, "baseline_legacy.npz"))


@pytest.fixture(scope="session")
def delight_model(weights_file):

    """the rebuilt Keras 3 model with the trained weights loaded"""

    import importlib.util
    modelpy = os.path.join(ROOT, "delight", "delight", "model.py")
    spec = importlib.util.spec_from_file_location("delight_model_undertest", modelpy)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_delight_model(weights_file)


def model_predict(model, Xpr):

    """run the model on a (batch, 30, 30, nlevels) preprocessed cube"""

    inputs = [Xpr[:, :, :, i][..., np.newaxis].astype("float32")
              for i in range(Xpr.shape[3])]
    return model.predict(inputs, verbose=0)
