"""Check that a missing or unusable image folder is reported clearly.

check_missing() returns False and never adds a "filename" column when it finds
nothing usable.  Nothing acts on that return value, so before these guards the
problem surfaced much later as an AttributeError on row.filename, or an
IsADirectoryError on an empty filename, raised from inside a dataframe apply in
get_pix_coords.

These tests need no real images: check_missing() parses coordinates out of the
file names and never opens them, so empty files with the right names are enough.
"""

import os

import numpy as np
import pytest

# ZTF21aacusce and ZTF21abtxglm, the first two objects of the test sample
OIDS = np.array(["ZTF21aacusce", "ZTF21abtxglm"])
RAS = np.array([20.404016916666667, 15.187699694736844])
DECS = np.array([-22.60269873333333, -21.432253836842104])


def make_client(datadir):
    from delight.delight import Delight
    return Delight(str(datadir), OIDS, RAS, DECS)


def touch_cutout(datadir, ra, dec):

    """create an empty file named the way a downloaded cutout would be"""

    name = "stack_r_ra%.6f_dec%.6f_arcsec120.fits" % (ra, dec)
    open(os.path.join(str(datadir), "fits", name), "w").close()


def test_empty_folder_reports_itself(tmp_path):

    """the case a first-time user hits: nothing downloaded yet"""

    client = make_client(tmp_path)
    assert client.check_missing() is False

    with pytest.raises(ValueError) as excinfo:
        client.get_pix_coords()
    message = str(excinfo.value)
    assert "No usable image files" in message
    assert "download()" in message
    # the message must name the folder it actually looked in
    assert client.downloadfolder in message


def test_folder_without_cutouts_reports_itself(tmp_path):

    """files are present, but none are PanSTARRS cutouts"""

    client = make_client(tmp_path)
    open(os.path.join(client.downloadfolder, "notes.txt"), "w").close()

    assert client.check_missing() is False
    with pytest.raises(ValueError):
        client.get_pix_coords()


def test_partially_downloaded_sample_names_the_gap(tmp_path):

    """some objects have an image and some do not"""

    client = make_client(tmp_path)
    touch_cutout(tmp_path, RAS[0], DECS[0])

    # a catalogue exists, so check_missing succeeds and blanks the unmatched one
    assert client.check_missing() is True
    assert (client.df.filename == "").sum() == 1

    with pytest.raises(ValueError) as excinfo:
        client.get_pix_coords()
    message = str(excinfo.value)
    assert "1 of 2 objects" in message
    # and it must say which one is missing, not just how many
    assert "ZTF21abtxglm" in message


def test_complete_sample_passes_the_guard(tmp_path):

    """the guard must not reject a sample that does have all its images

    These files are empty, so astropy rejects them as corrupt once it opens
    them. That specific failure is the point: reaching fits.open at all proves
    the guard passed the sample through rather than stopping it.
    """

    client = make_client(tmp_path)
    for ra, dec in zip(RAS, DECS):
        touch_cutout(tmp_path, ra, dec)

    assert client.check_missing() is True
    assert (client.df.filename == "").sum() == 0

    with pytest.raises(OSError, match="Empty or corrupt FITS file"):
        client.get_pix_coords()
