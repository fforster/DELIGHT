"""End-to-end regression test: the modernized pipeline against the original.

The golden file was produced by the pre-migration code running under
TensorFlow 2.15 / Keras 2 / NumPy 1.26 / pandas 2.1, over the 32 objects in
data/testcoords.csv.  This test reruns the whole pipeline on the same fits
files and compares stage by stage, so a discrepancy is attributed to
preprocessing, source extraction, the network or the WCS rather than blamed on
whichever one is most convenient.

It is offline: it needs the fits files in data/fits but never downloads.
"""

import os

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT

# quantity -> (stage, tolerance).  The bands differ by stage on purpose: the
# network should agree to float32 noise, while sep- and WCS-derived quantities
# are allowed to move slightly with their library versions.
TOLERANCES = [
    ("X", "preprocessing", 0.0),
    ("Xpr", "preprocessing", 0.0),
    ("xSN", "source extraction", 1e-3),
    ("ySN", "source extraction", 1e-3),
    ("dx_sex", "source extraction", 1e-3),
    ("dy_sex", "source extraction", 1e-3),
    ("y_raw", "neural network", 1e-4),
    ("dxdy_rotflip", "neural network", 1e-4),
    ("dx_delight", "neural network", 1e-4),
    ("dy_delight", "neural network", 1e-4),
    ("std_delight", "neural network", 1e-4),
    ("ra_delight", "coordinates", 1e-7),
    ("dec_delight", "coordinates", 1e-7),
    ("ra_sex", "coordinates", 1e-7),
    ("dec_sex", "coordinates", 1e-7),
    ("hostsize", "host size", 1e-3),
    ("hostsep", "host size", 1e-3),
    ("mindistsize", "host size", 1e-3),
]

DATADIR = os.path.join(ROOT, "data")


@pytest.fixture(scope="module")
def pipeline():

    """run the current code over the test sample, offline"""

    fitsdir = os.path.join(DATADIR, "fits")
    if not os.path.isdir(fitsdir) or not os.listdir(fitsdir):
        pytest.skip("no fits files in %s" % fitsdir)

    from delight.delight import Delight

    df = pd.read_csv(os.path.join(DATADIR, "testcoords.csv"))
    client = Delight(DATADIR, df.oid.values, df.ra.values, df.dec.values)

    if not client.check_missing():
        pytest.skip("no usable fits files")
    if (client.df.dist > 0.1).any():
        pytest.skip("some objects have no local fits file")

    client.get_pix_coords()
    client.compute_multiresolution(5, False, True, False)
    client.load_model()
    client.preprocess()
    client.predict()
    for oid in client.df.index:
        client.get_hostsize(oid, doplot=False)

    results = {"oids": np.asarray(client.oids, dtype="U32"),
               "X": np.asarray(client.X, dtype="float64"),
               "Xpr": np.asarray(client.Xpr, dtype="float64"),
               "y_raw": np.asarray(client.y_pred_raw, dtype="float64"),
               "dxdy_rotflip": np.stack(
                   client.df["dxdy_delight_rotflip"].to_numpy()).astype("float64")}
    for name, _, _ in TOLERANCES:
        if name not in results and name in client.df:
            results[name] = client.df[name].to_numpy(dtype="float64")
    return results


def test_same_objects(pipeline, baseline):
    assert np.array_equal(pipeline["oids"], baseline["oids"])


@pytest.mark.parametrize("name,stage,tol", TOLERANCES,
                         ids=[t[0] for t in TOLERANCES])
def test_matches_baseline(pipeline, baseline, name, stage, tol):

    """one quantity, against the pre-migration reference"""

    if name not in baseline.files:
        pytest.skip("%s not in the golden file" % name)
    assert name in pipeline, "%s missing from the run" % name

    old = np.asarray(baseline[name], "float64")
    new = np.asarray(pipeline[name], "float64")
    assert new.shape == old.shape

    diff = np.abs(new - old)
    worst = pipeline["oids"][int(np.argmax(
        diff.reshape(diff.shape[0], -1).max(axis=1) if diff.ndim > 1 else diff))]
    assert diff.max() <= tol, \
        "%s (%s): max abs diff %.3e > %.1e, worst on %s" % (
            name, stage, diff.max(), tol, worst)


def test_matches_saved_dataframe(pipeline):

    """cross-check against the pickled dataframe the library writes

    This is an independent witness to the npz golden file, and it also proves
    the pandas 3 round-trip of the object columns still works.
    """

    pkl = os.path.join(DATADIR, "coords_all_data_nlevels5_maskFalse_objectsTrue.pkl")
    if not os.path.exists(pkl):
        pytest.skip("no saved dataframe to compare against")

    df = pd.read_pickle(pkl)
    for name in ("dx_delight", "dy_delight", "std_delight", "ra_delight", "dec_delight"):
        stored = df.loc[list(pipeline["oids"]), name].to_numpy(dtype="float64")
        assert np.abs(stored - pipeline[name]).max() < 1e-4, name
