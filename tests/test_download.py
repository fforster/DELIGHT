"""Check the HiPS2fits download path.

Deselected by default because it hits the CDS HiPS service.  Run it with

    pytest -m network
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

pytestmark = pytest.mark.network


def test_download_one_field():

    """fetch a single PanSTARRS r band cutout and read it back"""

    from astropy.io import fits

    from delight.delight import Delight

    datadir = tempfile.mkdtemp(prefix="delight-download-")
    try:
        # SN2004aq, comfortably inside the PanSTARRS footprint
        client = Delight(datadir, np.array(["SN2004aq"]),
                         np.array([179.61354]), np.array([10.01792]))
        client.download()

        downloaded = os.listdir(client.downloadfolder)
        assert downloaded, "nothing was downloaded"

        data = fits.open(os.path.join(client.downloadfolder, downloaded[0]))[0].data
        assert data.shape == (480, 480)
        assert np.isfinite(data).any()
    finally:
        shutil.rmtree(datadir, ignore_errors=True)
