
<img src="http://alerce.science/static/img/alerce_logo.cc79ccea2406.png" alt="drawing" width="300"/>

The **Deep Learning Identification of Galaxy Hosts in Transients (DELIGHT, Förster et al. 2022)** is a library created by the [ALeRCE broker](http://alerce.science) to automatically identify host galaxies of transient candidates using multi-resolution images and a convolutional neural network (you can test it with our [example notebook](https://nbviewer.org/github/fforster/DELIGHT/blob/main/notebook/Delight_example_notebook.ipynb)).

The library has a class with several methods that allow you to get the most likely host coordinates starting from given transient coordinates. In order to do this, the delight object needs a list of object identifiers and coordinates (`oid, ra, dec`). With this information, it downloads PanSTARRS images centered around the position of the transients (2 arcmin x 2 arcmin), gets their WCS solutions, creates the multi-resolution images, does some extra preprocessing of the data, and finally predicts the position of the hosts using a multi-resolution image and a convolutional neural network. It can also estimate the host's semi-major axis if requested taking advantage of the multi-resolution images.

Note that DELIGHT's prediction time is currently dominated by the time to download PanSTARRS images using the panstamps service. In the future, we expect that there will be services that directly provide multi-resolution images, which should be more lightweight with no significant loss of information. Once these images are obtained, the processing times are only milliseconds per host.

---

*Classes*:

* **Delight**: the main class containing the methods to predict host galaxy positions starting from transient coordinates

*Methods* (most important):

* **init**: class constructor, it requires a list of object identifiers, a list of right ascensions, and a list of declinations
* **download**: downloads PanSTARRS fits files using the [panstamps](https://panstamps.readthedocs.io/en/master/) servive.
* **get_pix_coords**: gets the WCS solution in the fits files to move from pixel to celestial coordinates.  
* **compute_multiresolution**: transform the PanSTARRS images to multi-resolution images
* **load_model**: load DELIGHT's tensorflow model
* **predict**: predict the host positions
* **plot_host**: plot the original host image, the multi-resolution images, and the transient and predicted host position
* **get_hostsize**: estimate the host semi-major axis
* **save**: save the resulting dataframe
* **load**: load the resulting dataframe

*Dependencies*:

* xarray (`python -m pip install xarray`)
* astropy (`pip install astropy`)
* sep (`pip install sep`)
* tensorflow (https://www.tensorflow.org/install/pip, `pip install tensorflow`)
* pantamps (`pip install panstamps`)

--- 
**DELIGHT's multi-resolution images and prediction vector:**

<img src="https://github.com/fforster/DELIGHT/blob/main/figures/multi-resolution.png" alt="drawing" width="600"/>

**DELIGHT's neural network architecture:**

<img src="https://github.com/fforster/DELIGHT/blob/main/figures/delight_architecture.png" alt="drawing" width="600"/>
