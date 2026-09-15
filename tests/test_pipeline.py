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

# quantity -> (stage, tolerance, unit).
#
# Tolerances are absolute, applied to the maximum absolute difference, and each
# is in the unit of its own quantity -- they are not comparable across stages.
# The images are PanSTARRS cutouts at 0.25 arcsec per pixel (Delight.pixscale),
# so 1e-3 px is 0.25 mas, 1e-4 px is 25 uas, 1e-7 deg is 0.36 mas and
# 1e-3 arcsec is 1 mas.
#
# The bands differ by stage on purpose: the network should agree to float32
# noise, while sep- and WCS-derived quantities are allowed to move slightly
# with their library versions.
TOLERANCES = [
    # min-max normalised intensity in [0, 1], so dimensionless, and compared
    # exactly: these stages are pure numpy and xarray and should not move at all
    ("X", "preprocessing", 0.0, ""),
    ("Xpr", "preprocessing", 0.0, ""),
    # pixel coordinates and offsets on the 480x480 cutout
    ("xSN", "source extraction", 1e-3, "px"),
    ("ySN", "source extraction", 1e-3, "px"),
    ("dx_sex", "source extraction", 1e-3, "px"),
    ("dy_sex", "source extraction", 1e-3, "px"),
    # the network predicts offsets in cutout pixels: dx_delight is added to xSN
    # before wcs.pixel_to_world. std_delight is an rms over those offsets
    ("y_raw", "neural network", 1e-4, "px"),
    ("dxdy_rotflip", "neural network", 1e-4, "px"),
    ("dx_delight", "neural network", 1e-4, "px"),
    ("dy_delight", "neural network", 1e-4, "px"),
    ("std_delight", "neural network", 1e-4, "px"),
    ("ra_delight", "coordinates", 1e-7, "deg"),
    ("dec_delight", "coordinates", 1e-7, "deg"),
    ("ra_sex", "coordinates", 1e-7, "deg"),
    ("dec_sex", "coordinates", 1e-7, "deg"),
    # get_hostsize multiplies both of these by pixscale before storing them
    ("hostsize", "host size", 1e-3, "arcsec"),
    ("hostsep", "host size", 1e-3, "arcsec"),
    # separation divided by semi-major axis: a ratio, not an angle, so it is
    # kept out of the arcsec group above rather than implying a unit it lacks
    ("mindistsize", "host size ratio", 1e-3, ""),
]

DATADIR = os.path.join(ROOT, "data")


@pytest.fixture(scope="module")
def pipeline_client():

    """run the current code over the test sample, offline

    Returns the Delight object itself, so tests that need its methods rather
    than just its numbers can use it.
    """

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
    for name, _, _, _ in TOLERANCES:
        if name not in results and name in client.df:
            results[name] = client.df[name].to_numpy(dtype="float64")
    return client, results


@pytest.fixture(scope="module")
def pipeline(pipeline_client):

    """the numbers produced by that run"""

    return pipeline_client[1]


def test_same_objects(pipeline, baseline):
    assert np.array_equal(pipeline["oids"], baseline["oids"])


@pytest.mark.parametrize("name,stage,tol,unit", TOLERANCES,
                         ids=[t[0] for t in TOLERANCES])
def test_matches_baseline(pipeline, baseline, name, stage, tol, unit):

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
        "%s (%s): max abs diff %.3e > %.1e %s, worst on %s" % (
            name, stage, diff.max(), tol, unit or "(dimensionless)", worst)


def test_matches_saved_dataframe(pipeline):

    """cross-check against the pickled dataframe the library writes

    This is an independent witness to the npz golden file, and it also proves
    the pandas 3 round-trip of the object columns still works.
    """

    pkl = os.path.join(DATADIR, "coords_all_data_nlevels5_maskFalse_objectsTrue.pkl")
    if not os.path.exists(pkl):
        pytest.skip("no saved dataframe to compare against")

    try:
        df = pd.read_pickle(pkl)
    except AttributeError as error:
        # The dataframe stores astropy WCS and SkyCoord objects in its cells, and
        # a pickle written by one astropy major version cannot always be read by
        # another. That is astropy's pickle format, not something this package
        # controls, so skip rather than fail; save() and load() round-tripping on
        # a single stack is what matters and is covered elsewhere.
        pytest.skip("saved dataframe was written by a different astropy: %s" % error)

    # reuse the per-quantity bands rather than one number for all of them:
    # these columns are a mix of pixels and degrees, and a pixel tolerance
    # applied to a right ascension would be looser than a whole pixel
    bands = {name: (tol, unit) for name, _, tol, unit in TOLERANCES}

    for name in ("dx_delight", "dy_delight", "std_delight", "ra_delight", "dec_delight"):
        tol, unit = bands[name]
        stored = df.loc[list(pipeline["oids"]), name].to_numpy(dtype="float64")
        diff = np.abs(stored - pipeline[name]).max()
        assert diff <= tol, "%s: max abs diff %.3e > %.1e %s" % (name, diff, tol, unit)


def test_save_load_round_trip(pipeline_client, tmp_path):

    """save() then load() returns the same numbers on one stack

    The dataframe holds astropy WCS and SkyCoord objects and (8, 2) arrays in
    object columns, so this is really a check that pandas 3 pickles and restores
    those cells intact. Writing goes to a temporary directory rather than the
    repository's data directory.
    """

    client, results = pipeline_client

    original_datadir = client.datadir
    original_df = client.df.copy()
    try:
        client.datadir = str(tmp_path)
        client.save()

        client.df = pd.DataFrame()      # prove load() really repopulates it
        client.load()

        for name in ("dx_delight", "dy_delight", "std_delight",
                     "ra_delight", "dec_delight", "hostsize"):
            restored = client.df.loc[list(results["oids"]), name].to_numpy(dtype="float64")
            assert np.array_equal(restored, results[name]), name

        # the object columns, which are the ones pickling could quietly mangle
        rotflip = np.stack(client.df["dxdy_delight_rotflip"].to_numpy()).astype("float64")
        assert np.array_equal(rotflip, results["dxdy_rotflip"])
        assert client.df.wcs.iloc[0].pixel_to_world(0, 0) is not None
    finally:
        client.datadir = original_datadir
        client.df = original_df
