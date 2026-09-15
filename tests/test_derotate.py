"""Check that derotate() is matched to the model's variant ordering.

The network sees each image under eight dihedral transforms and emits a
prediction for each.  Delight.derotate() maps those eight predictions back into
the original frame *by index*, so if the model's variant ordering and
derotate's row ordering ever drift apart, the predictions become wrong with no
error raised.

The invariant that catches this is that the eight derotated predictions must
agree with each other: they are eight views of the same host position.  A
permuted ordering destroys that agreement, and by a wide margin.
"""

import numpy as np

NVARIANTS = 8

# All three are in pixels on the 480x480 PanSTARRS cutout, which is 0.25 arcsec
# per pixel. The eight derotated predictions of one object typically scatter by
# ~2 px, while a wrong ordering scatters them by more than ten.
MAX_SPREAD = 8.0            # px, worst single object
MAX_MEAN_SPREAD = 4.0       # px, averaged over the sample
MIN_PERMUTED_SPREAD = 8.0   # px, a permuted ordering must exceed this


def spread(dxdy):

    """mean scatter of the per-variant predictions, in pixels

    Parameters
    ----------
    dxdy : numpy array
       array of shape (nobjects, nvariants, 2)
    """

    mean = dxdy.mean(axis=1)
    return np.sqrt(((dxdy - mean[:, np.newaxis, :]) ** 2).sum(axis=2)).mean(axis=1)


def derotate(client, y_raw):
    return client.derotate(y_raw)


def test_derotate_reproduces_golden(baseline):

    """derotate() applied to the raw output gives the stored per-variant array"""

    from delight.delight import Delight

    client = Delight.__new__(Delight)     # derotate needs no instance state
    got = client.derotate(baseline["y_raw"])
    # atol in pixels; this is a pure sign and axis swap, so it is exact
    assert np.allclose(got, baseline["dxdy_rotflip"], atol=1e-9)


def test_variants_agree(baseline):

    """the eight derotated predictions describe the same host position"""

    scatter = spread(baseline["dxdy_rotflip"])
    assert scatter.mean() < MAX_MEAN_SPREAD, "mean scatter %.2f px" % scatter.mean()
    assert scatter.max() < MAX_SPREAD, "worst scatter %.2f px" % scatter.max()


def test_wrong_ordering_would_be_caught(baseline):

    """a permuted variant ordering must break the agreement above

    This is what makes test_variants_agree meaningful rather than vacuous: it
    shows the invariant is actually sensitive to the ordering it is protecting.
    """

    from delight.delight import Delight

    client = Delight.__new__(Delight)
    y_raw = baseline["y_raw"]
    perobject = y_raw.reshape(-1, NVARIANTS, 2)

    rs = np.random.RandomState(0)
    worst = np.inf
    for _ in range(50):
        permutation = rs.permutation(NVARIANTS)
        if (permutation == np.arange(NVARIANTS)).all():
            continue
        permuted = perobject[:, permutation, :].reshape(-1, 2 * NVARIANTS)
        worst = min(worst, spread(client.derotate(permuted)).mean())

    assert worst > MIN_PERMUTED_SPREAD, \
        "a permuted ordering scattered by only %.2f px, so the check is too weak" % worst


def test_model_variant_order_matches_derotate():

    """delight.model.VARIANTS is the order derotate() expects"""

    from delight.delight import VARIANTS

    assert len(VARIANTS) == NVARIANTS
    # four rotations without a flip, then the same four with one
    assert VARIANTS == ((False, 0), (False, 1), (False, 2), (False, 3),
                        (True, 0), (True, 1), (True, 2), (True, 3))
