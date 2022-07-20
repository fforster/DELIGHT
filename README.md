Francisco Förster et al. 2022

The Deep Learning Identification of Galaxy Hosts in Transients (DELIGHT, Förster et al. 2022) is a library created by the ALeRCE broker to automatically identify host galaxies of transient candidates using multi-resolution images and a convolutional neural network.

The library has a class with different subroutines that allows you to get the most likely host coordinates starting from given transient coordinates.

In order to do this, the delight object needs a list of object identifiers and coordinates (oid, ra, dec). With this information, it downloads PanSTARRS images centered around the position of the transients (2 arcmin x 2 arcmin), gets their WCS solutions, creates the multi-resolution images, does some extra preprocessing of the data, and finally predicts the position of the hosts using a multi-resolution image and a convolutional neural network. It can also estimate the host's semi-major axis if requested taking advantage of the multi-resolution images.

Note that DELIGHT's prediction time is currently dominated by the time to download PanSTARRS images using the panstamps service. In the future, we expect that there will be services that directly provide multi-resolution images, which should be more lightweight with no significant loss of information.


*Classes*:

**Delight**


*Dependencies*:

* xarray (!python -m pip install xarray)
* astropy (!pip install astropy)
* sep (!pip install sep)
* tensorflow (https://www.tensorflow.org/install/pip, !pip install tensorflow)
* pantamps (!pip install panstamps)