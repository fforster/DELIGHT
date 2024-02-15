import os
import re
import subprocess
import numpy as np
import pandas as pd
import pkg_resources

import warnings

import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from astropy.visualization import ZScaleInterval, ImageNormalize, MinMaxInterval, PercentileInterval, LogStretch, LinearStretch
from astropy import units as u
from astropy.coordinates import SkyCoord

#import fitsio
import sep
from astropy.wcs import WCS
from astropy.io import fits

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm

import tensorflow as tf
import panstamps # not really used here, but if we cannot import it is likely that it is not installed

class Delight(object):

    """collection of transient locations and images"""

    def __init__(self, datadir, oids, ras, decs):

        """Create TransientSet object

        Parameters
        ----------
        datadir : String
           directory where the image files are located
        oids : numpy array
           vector with transient ids
        ras : numpy array
           vector with right ascensions
        decs : numpy array
           vector with declinations
        """
        
        self.datadir = datadir
        self.df = pd.DataFrame({"oid": oids, "ra": ras, "dec": decs}).set_index("oid", drop=True)
        self.sn_coords = SkyCoord(self.df.ra.to_numpy(), self.df.dec.to_numpy(), unit=(u.deg, u.deg))
        
        # download directory
        self.downloadfolder = os.path.join(self.datadir, "fits")

        if not os.path.exists(self.downloadfolder):
            os.makedirs(self.downloadfolder)

        # images directory
        self.imagefolder = os.path.join(self.datadir, "images")

        if not os.path.exists(self.imagefolder):
            os.makedirs(self.imagefolder)

        # pixel scale
        self.pixscale = 0.25

            
    def check_missing(self):

        """check missing files to download"""

        """
        Parameters
        ----------
        None
        """

        
        # check existing files
        files = os.listdir(self.downloadfolder)

        if files == []:
            return False
        
        filters = []
        matchedfiles = []
        ras = []
        decs = []
        skycoords = []
        for f in files:
            if re.match('stack_r_ra(.*?)_dec(.*)_arcsec120.*fits', f):
                flt = 'r'
                ra, dec = re.findall('stack_r_ra(.*?)_dec(.*)_arcsec120.*fits', f)[0]
                filters.append(flt)
                ras.append(float(ra))
                decs.append(float(dec))
                matchedfiles.append(f)
        dfpanstamps = pd.DataFrame({"filename": matchedfiles, "filters": filters, "panstamps_ra": ras, "panstamps_dec": decs})
        self.panstamps_coords = SkyCoord(dfpanstamps.panstamps_ra.to_numpy(), dfpanstamps.panstamps_dec.to_numpy(), unit=(u.deg, u.deg))

        # find xmatches between requested SNe and available panstamp files
        idx, d2d, d3d = self.sn_coords.match_to_catalog_sky(self.panstamps_coords)
        self.df["dist"] = np.array([float(dist / u.arcsec) for dist in d2d])
        self.df["filename"] = dfpanstamps.filename.to_numpy()[idx]
        self.df.loc[self.df.dist > 0.1, "filename"] = ""

        return True
        
    def download(self, width=2, overwrite=False):

        """Download missing data"""

        """
        Parameters
        ----------
        width : integer
           the width of the image in arcmin (default 2)
        overwrite : bool
           whether to overwrite the images
        """

        check = self.check_missing()
        if check:
            print(f"Downloading {(self.df.dist > 0.1).sum()} missing files.")

        # download missing files
        for idx, row in self.df.iterrows():
            if "dist" in self.df:
                if row.dist > 0.1 or overwrite:
                    command = 'panstamps -f --width=%i --filter=r --downloadFolder=%s stack %s %s' % (width, self.downloadfolder, row.ra, row.dec)
                    print(command)
                    output = subprocess.check_output(command, shell=True)
                    if len(output) == 0:
                        print(f"   WARNING: object {idx} ({row.ra} {row.dec}) cannot be downloaded, probably outside PS1 footprint. Please remove {idx} from your sample.")
            else:
                command = 'panstamps -f --width=%i --filter=r --downloadFolder=%s stack %s %s' % (width, self.downloadfolder, row.ra, row.dec)
                print(command)
                output = subprocess.check_output(command, shell=True)
                if len(output) == 0:
                    print(f"   WARNING: object {idx} ({row.ra} {row.dec}) cannot be downloaded, probably outside PS1 footprint. Please remove {idx} from the sample.") 
        self.check_missing()

        
    def get_pix_coords(self):

        """get WCS information and compute SN pixel coordinates"""


        """
        Parameters
        ----------
        None
        """
        
        # get wcs
        print("Loading WCS information")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # code that produces a warning
            self.df["wcs"] = self.df.apply(lambda row: WCS(fits.open(os.path.join(self.downloadfolder, row.filename), output_verify='silent_fix')[0].header), axis=1)        

        self.df["sn_coords"] = self.df.apply(lambda row: SkyCoord(row.ra, row.dec, unit='deg'), axis=1)
        self.df["xSN"] = self.df.apply(lambda row: row.wcs.world_to_pixel(row.sn_coords)[0], axis=1)
        self.df["ySN"] = self.df.apply(lambda row: row.wcs.world_to_pixel(row.sn_coords)[1], axis=1)
        self.df["dx"] = 1
        self.df["dy"] = 1


    def get_objects(self, data):
        
        """get SExtractor objects"""
        
        """
        Parameters
        ----------
        data : numpy array
           numpy array with 2D image
        """

        # measure a spatially varying background on the image
        bkg = sep.Background(data, bw=128, bh=128, fw=3, fh=3)
        bkg_image = bkg.back()
        
        # subtract the background
        data_sub = data - bkg
        
        # extract objects
        objects = sep.extract(data_sub, 1.5, err=bkg.globalrms)
        
        # create mask
        mask = np.zeros_like(data_sub, dtype=bool)
        sep.mask_ellipse(mask, objects['x'], objects['y'], objects['a'], objects['b'], objects['theta'], r=3)
    
        return objects, mask

    
    def ell_dist(self, x, y, cx, cy, cxx, cxy, cyy):

        """elliptical distances"""
        
        """
        Parameters
        ----------
        x : float
           SExtractor source x parameter
        y : float
           SExtractor source y parameter
        cx : float
           SExtractor source cx parameter
        cy : float
           SExtractor source cy parameter
        cxx : float
           SExtractor source cxx parameter
        cxy : float
           SExtractor source cxy parameter
        cyy : float
           SExtractor source cyy parameter
        """

        return cxx * (x - cx)**2 + cxy * (y - cy) * (x - cx) + cyy * (y - cy)**2


    def plot_data(self, image, objects=None, ax=None):
    
        """plot one masked image with detected SExatractor objects if available"""

        """
        Parameters
        ----------
        image : numpy array
           numpy array with 2D image
        objects : pandas dataframe
           pandas dataframe with list of SExtractor detected sources (do not plot if None)
        ax : matplotlib axis
           axis where to plot (create new figure is None)
        """


        # plot background-subtracted image
        if ax is None:
            fig, ax = plt.subplots()
            
        image_masked = np.ma.masked_where((image <= 0), image)
        norm = ImageNormalize(image_masked[image_masked > 0], interval=ZScaleInterval(), vmin=np.min(image_masked), vmax= np.percentile(image_masked, 99))
        im = ax.imshow(image_masked, interpolation='nearest', cmap='viridis', origin='lower', norm=norm)
    
        # plot an ellipse for each object
        if objects is not None:
    
            first = True
            for idx, row in objects.sort_values('ndist').iterrows():
    
                e = Ellipse(xy=(row.x - 0.5, row.y - 0.5),
                        width=6*row.a, height=6*row.b,
                        angle=row.theta * 180. / np.pi)
                e.set_facecolor('none')
                if first:
                    e.set_edgecolor('k')
                    e.set_linewidth(3)
                    first = False
                    xc = image.shape[0]/2
                    ax.arrow(xc, xc, row.x - xc, row.y - xc, color='k', width=0.3, head_width=3, length_includes_head=True)
                else:
                    e.set_edgecolor('white')
                ax.add_artist(e)
        ax.set_xlim(-0.5, image.shape[1] - 0.5)



    def get_data(self, oid, filename, dx=None, dy=None, nlevels=5, domask=False, doobject=False, doplot=False):

        """get data and convert to multiresolution images"""

        """
        Parameters
        ----------
        oid : String
           object identifier
        filename : String
           filename associated to given oid
        dx : float 
           offset vector x component
        dy : float
           offset vector y component
        nlevels : integer
           number of levels to use in the multi-resolution representation
        domask : bool
           whether to apply a median absolute deviation mask
        doobject : bool
           whether to apply a SExtractor sources mask
        doplot : bool
           whether to plot the data
        """

        if domask and doobject:
            print(f"WARNING: domask and doobject cannot be true simultaneously.")
            return
        
        #data = fitsio.read(filename)
        data = fits.open(filename)[0].data.byteswap().newbyteorder()
        data = np.nan_to_num(data, 0)
        data = data# * 1.0
        if (np.sum(data == 0) == data.shape[0] * data.shape[1]):
            print(f"   WARNING: Fits file {filename} has only zeros. Please remove from your sample.")
            return None, None, None, None, None

        # show the image
        if domask:
            med = np.median(data)
            data = data - med
            mad = np.median(np.abs(data))
            mask = (data > 3 * mad)
        elif doobject:
            objects, mask = self.get_objects(data)
            if len(objects) == 0:
                print(f"   WARNING: no objects detected in {filename}. Please remove from your sample.")
                return None, None, None, None, None
            objects = pd.DataFrame(data = objects, columns = ['thresh', 'npix', 'tnpix', 'xmin', 'xmax', 'ymin', 'ymax', 'x', 'y', 'x2', 'y2', 'xy', 'errx2', 'erry2', 'errxy', 'a', 'b', 'theta', \
        'cxx', 'cyy', 'cxy', 'cflux', 'flux', 'cpeak', 'peak', 'xcpeak', 'ycpeak', 'xpeak', 'ypeak', 'flag'])
            objects['cx'] = objects['cy'] = data.shape[0] / 2.
            try:
                objects["ndist"] = objects.apply(lambda row: 
                                                 self.ell_dist(row.x, row.y, row.cx, row.cy, row.cxx, row.cxy, row.cyy), axis=1)
                objects = objects.sort_values("ndist")
            except:
                print(f"   WARNING: no objects in {filename}. Please remove from your sample.")
                return None, None, None, None, None
        else:
            objects = None
            mask = np.ones_like(data, dtype=bool)
        
        # comment this line if you don't want to replace values below the mask
        if np.sum(mask) > 0:
            data = (data - data[mask].min()) / (data[mask].max() - data[mask].min()) 
        else:
            data = (data - data.min()) / (data.max() - data.min())
        
        # auxiliary array set to zero outside the mask
        aux = np.array(data * mask)
        
        # plot original
        if doplot:
            fig, ax = plt.subplots(ncols=2, figsize=(30, 15))
            if doobject:
                obj = objects.copy()
                center = int(aux.shape[0]/2)
                delta = int(aux.shape[0]/2.)
                for key in ['x', 'y']:
                    obj[key] = obj[key] - (center - delta)
                for key in ['a', 'b']:
                    obj[key] = obj[key]
            else:
                obj = None
            #print("data")
        
            norm = ImageNormalize(data, interval=ZScaleInterval())
            ax[0].imshow(data, origin='lower', cmap='viridis', norm=norm)
            ax[0].arrow(data.shape[0]/2, data.shape[0]/2, dx, dy, color='r', width=0.3, head_width=5, length_includes_head=True)
        
            #print("aux")
            self.plot_data(aux, obj, ax=ax[1])
            ax[0].axis('off')
            ax[1].axis('off')
            ax[1].arrow(data.shape[0]/2, data.shape[0]/2, dx, dy, color='r', width=0.3, head_width=5, length_includes_head=True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.imagefolder, "%s_objects.png" % (oid.replace(" ", ""))))        
            plt.show()
        

        # multi resolution images
        if doplot:
            fig, ax = plt.subplots(nrows=nlevels, ncols=8, figsize=(nlevels * 6, 16))
        
        # hierarchical data
        delta = int(aux.shape[0]/2**nlevels)
        datah = np.zeros((nlevels, 2 * delta, 2 * delta))
        
        # iterate each level
        for exp in range(nlevels):
            factor = 2**exp
            a = xr.DataArray(aux, dims=['x', 'y'])
            c = a.coarsen(x=factor, y=factor).median()
            center = int(c.shape[0]/2)
            image = c[center-delta: center+delta, center-delta: center+delta]
            datah[exp] = image
            if doplot:
                if doobject:
                    obj = objects.copy()
                    for key in ['x', 'y']:
                        obj[key] = obj[key] / factor - (center - delta)
                    for key in ['a', 'b']:
                        obj[key] = obj[key] / factor
                else:
                    obj = None
                for i in range(2):
                    for j in range(4):
                        a = xr.DataArray(aux, dims=['x', 'y'])
                        c = a.coarsen(x=factor, y=factor).median()
                        center = int(c.shape[0]/2)
                        image = c[center-delta: center+delta, center-delta: center+delta]
                        idx = i * 4 + j
                        #ax[exp, idx].plot([delta, delta + dx / factor], [delta, delta + dy / factor], c='k', lw=2)
                        ax[exp, idx].arrow(delta, delta, dx / factor, dy / factor, color='r', width=0.3, head_width=3, length_includes_head=True)
        
                        ax[exp, idx].axis('off')
                        ax[exp, idx].set_xlim(-0.5, image.shape[1] - 0.5)
                        ax[exp, idx].set_ylim(-0.5, image.shape[1] - 0.5)

                        if (i != 0) or (j != 0):
                            obj = None
                        self.plot_data(image, obj, ax=ax[exp, idx])
                        aux = np.rot90(aux)
                        if dx is not None and dy is not None:
                            dxold = dx
                            dx = dy
                            dy = -dxold
                    aux = np.fliplr(aux)
                    dx = -dx
        if doplot:
            plt.savefig(os.path.join(self.imagefolder,
                                     "%s_nlevels%i.png" % (oid.replace(" ", ""), nlevels)))
            plt.show()
                
        if doobject:
            return datah, dict(objects), mask, objects.iloc[0].x - objects.iloc[0].cx, objects.iloc[0].y - objects.iloc[0].cy
        else:
            return datah, mask

        
    def compute_multiresolution(self, nlevels, domask, doobject, doplot):

        """Compute multi-resolution images"""

        """
        Parameters
        ----------
        nlevels : integer
           number of levels to use in the multi-resolution representation
        domask : bool
           whether to apply a median absolute deviation mask
        doobject : bool
           whether to apply a SExtractor sources mask
        doplot : bool
           whether to plot the data
        """
        
        self.nlevels = nlevels
        self.domask = domask
        self.doobject = doobject

        coordsdata = self.df.apply(lambda row: self.get_data(row.name, os.path.join(self.downloadfolder, str(row.filename)), row.dx, row.dy, self.nlevels, self.domask, self.doobject, doplot), axis=1, result_type='expand')

        if doobject:
            coordsdata.columns = ['data', 'objects', 'mask', 'dx_sex', 'dy_sex']
            self.df = pd.concat([self.df, coordsdata], axis=1)
        else:
            coordsdata.columns = ['data', 'mask']
            self.df = pd.concat([self.df, coordsdata], axis=1)

        # design matrix and values
        self.X = np.stack(self.df.data)
        self.y = np.array(list(zip(self.df.dx, self.df.dy)))

        
    def save(self):

        """save dataframe"""

        """
        Parameters
        ----------
        None
        """

        
        outputfile = os.path.join(self.datadir, "coords_all_data_nlevels%i_mask%s_objects%s.pkl" % (self.nlevels, self.domask, self.doobject))

        print(f"Saving data into {outputfile}")
        
        self.df.to_pickle(outputfile)


    def load(self):

        """save dataframe"""

        """
        Parameters
        ----------
        None
        """
        
        coordsfile = os.path.join(self.datadir, "coords_all_data_nlevels%i_mask%s_objects%s.pkl" % (self.nlevels, self.domask, self.doobject))

        print(f"Loading data from {coordsfile}")
        
        self.df = pd.read_pickle(coordsfile)
        
        
    def get_masked(self, filename, domask, doobject):

        """get masked data"""

        """
        Parameters
        ----------
        filename : String
           filename with associated fits file
        domask : bool
           whether to apply a median absolute deviation mask
        doobject : bool
           whether to apply a SExtractor sources mask
        """
        
        # read data
        #data = fitsio.read(os.path.join(self.downloadfolder, filename))
        data = fits.open(os.path.join(self.downloadfolder, filename))[0].data.byteswap().newbyteorder()
        data = np.nan_to_num(data, 0)
        data = data# * 1.0
    
        # show the image
        if domask:
            med = np.median(data)
            data = data - med
            mad = np.median(np.abs(data))
            mask = (data > 3 * mad)
        elif doobject:
            objects, mask = self.get_objects(data)
            if len(objects) == 0:
                print(filename)
            objects = pd.DataFrame(data = objects, columns = ['thresh', 'npix', 'tnpix', 'xmin', 'xmax', 'ymin', 'ymax', 'x', 'y', 'x2', 'y2', 'xy', 'errx2', 'erry2', 'errxy', 'a', 'b', 'theta', \
        'cxx', 'cyy', 'cxy', 'cflux', 'flux', 'cpeak', 'peak', 'xcpeak', 'ycpeak', 'xpeak', 'ypeak', 'flag'])
            objects['cx'] = objects['cy'] = data.shape[0] / 2.
            try:
                objects["ndist"] = objects.apply(lambda row: 
                                                 self.ell_dist(row.x, row.y, row.cx, row.cy, row.cxx, row.cxy, row.cyy), axis=1)
                objects = objects.sort_values("ndist")
            except:
                print(filename)
                return None, None, None, None, None
        else:
            objects = None
            mask = np.ones_like(data, dtype=bool)
        
        # comment this line if you don't want to replace values below the mask
        if np.sum(mask) > 0:
            data = (data - data[mask].min()) / (data[mask].max() - data[mask].min()) 
        else:
            data = (data - data.min()) / (data.max() - data.min())
        
        # auxiliary array set to zero outside the mask
        aux = np.array(data * mask)
        
        return aux

    
    def plot_host(self, oid):

        """plot multi resolution image of host given oid"""

        """
        Parameters
        ----------
        oid : String
           object identifier
        """
        
        cmap = cm.get_cmap('viridis_r')

        idx_host = self.df.index.get_loc(oid)
        dxpred = self.df.loc[oid].dx_delight
        dypred = self.df.loc[oid].dy_delight
        dxdypred_all = self.df.loc[oid].dxdy_delight_rotflip
        snsize = 500
            
        fig, ax = plt.subplots(figsize=(30, 19))
        ax.axis('off')
        inset = ax.inset_axes([0, 1/3, 1/3, 2/3])
        image = fits.open(os.path.join(self.downloadfolder, self.df.loc[oid].filename))[0].data
        interval = ZScaleInterval(contrast=0.1)
        vmin, vmax = interval.get_limits(image[image>0])
        norm = ImageNormalize(vmin=vmin, vmax=vmax)#, stretch=SqrtStretch())
        image_masked = np.ma.masked_where((image <= 0), image)
        inset.imshow(image_masked, cmap=cmap, norm=norm, origin='lower', interpolation='nearest')
        nx = ny = image.shape[0]
        inset.text(nx * 0.01, ny * 0.99, oid, fontsize=30, c='k', va='top')
        inset.scatter(nx/2 - 0.5, nx/2 - 0.5, marker='*', c='white', s=snsize, zorder=1000)
        inset.arrow((nx - 1) / 2, (ny - 1) / 2, dxpred, dypred, color='white', width=3, head_width=9, length_includes_head=True)
        for dx, dy in dxdypred_all:
            inset.scatter((nx - 1) / 2 + dx, (ny - 1) / 2 + dy, lw=3, marker='.', s=10, color='none', edgecolor='violet')
        inset.scatter((nx - 1) / 2 + dxpred, (ny - 1) / 2 + dypred, lw=3, marker='o', s=200, color='none', edgecolor='white')
        inset.set_xticks([])
        inset.set_yticks([])
        
        inset = ax.inset_axes([1/3, 1/3, 1/3, 2/3])
        image = self.get_masked(self.df.loc[oid].filename, self.domask, self.doobject)
        interval = ZScaleInterval(contrast=0.2)
        vmin, vmax = interval.get_limits(image[image>0])
        norm = ImageNormalize(vmin=vmin, vmax=vmax)#, stretch=SqrtStretch())
        image_masked = np.ma.masked_where((image <= 0), image)
        inset.imshow(image_masked, cmap=cmap, norm=norm, origin='lower', interpolation='nearest')
        nx = ny = image.shape[0]
        inset.scatter(nx/2 - 0.5, nx/2 - 0.5, marker='*', c='white', s=snsize, zorder=1000)
        inset.arrow((nx - 1) / 2, (ny - 1) / 2, dxpred, dypred, color='white', width=3, head_width=9, length_includes_head=True)
        for dx, dy in dxdypred_all:
            inset.scatter((nx - 1) / 2 + dx, (ny - 1) / 2 + dy, lw=3, marker='.', s=10, color='none', edgecolor='violet')
        inset.scatter((nx - 1) / 2 + dxpred, (ny - 1) / 2 + dypred, lw=3, marker='o', s=200, color='none', edgecolor='white')
        inset.set_xticks([])
        inset.set_yticks([])
    
        inset = ax.inset_axes([2/3, 1/3, 1/3, 2/3])
        for i in [4, 3, 2, 1, 0]:
            image = self.X[idx_host, i, :, :]
            nx = ny = image.shape[0]
            #for j in range(nx + 1):
            #    inset.axvline(j - 0.5, c='gray', alpha=0.1)
            #    inset.axhline(j - 0.5, c='gray', alpha=0.1)
            interval = ZScaleInterval(contrast=0.25)
            if (image > 0).sum() > 0:
                vmin, vmax = interval.get_limits(image[image>0])
                norm = ImageNormalize(vmin=vmin, vmax=vmax)#, stretch=SqrtStretch())
                image_masked = np.ma.masked_where((image <= 0), image)
                inset.scatter(nx/2 - 0.5, nx/2 - 0.5, marker='*', c='white', s=snsize, zorder=1000)
                inset.imshow(image_masked, cmap=cmap, norm=norm, origin='lower', interpolation='nearest')
            else:
                inset.imshow(image_masked, origin='lower', interpolation='nearest')
    
            inset.arrow((nx - 1) / 2, (ny - 1) / 2, dxpred / 2**i, dypred / 2**i, color='white', width=1, head_width=3, length_includes_head=True)
            for dx, dy in dxdypred_all:
                inset.scatter((nx - 1) / 2 + dx / 2**i, (ny - 1) / 2 + dy / 2**i, lw=3, marker='.', s=10, color='none', edgecolor='violet')
            inset.scatter((nx - 1) / 2 + dxpred / 2**i, (ny - 1) / 2 + dypred / 2**i, lw=3, marker='o', s=200, color='none', edgecolor='white')#, width=0.2, head_width=0.5, length_includes_head            #inset.arrow(nx / 2, ny / 2, dxpred / 2**i, dypred / 2**i, color='r', width=0.1, head_width=0.2, length_includes_head=True)
            inset.set_xlim(-0.5, nx-0.5)
            inset.set_ylim(-0.5, nx-0.5)
            if i > 4:
                inset.axis('off')
            else: 
                inset.set_xticks([])
                inset.set_yticks([])
                inset.spines['bottom'].set_color('gray')
                inset.spines['top'].set_color('gray')
                inset.spines['left'].set_color('gray')
                inset.spines['right'].set_color('gray')
            if i > 0:
                inset = inset.inset_axes([0.25, 0.25, 0.5, 0.5])

        inset = ax.inset_axes([0, 0.05, 1, 1/3])
        image = np.concatenate([self.X[idx_host, i] for i in range(5)[::-1]], axis=1)
        interval = ZScaleInterval(contrast=0.25)
        if (image > 0).sum() > 0:
            vmin, vmax = interval.get_limits(image[image>0])
            norm = ImageNormalize(vmin=vmin, vmax=vmax)#, stretch=SqrtStretch())
            image_masked = np.ma.masked_where((image <= 0), image)
            inset.imshow(image_masked, cmap=cmap, norm=norm, origin='lower')    
        inset.set_xticks([])
        inset.set_yticks([])
        inset.text(nx * 0.01, ny * 0.06, "DELIGHT predicted host", c='k', fontsize=20)
        inset.text(nx * 0.01, ny * 0.01, "(white circle)", c='k', fontsize=20)
        nx = image.shape[0]
        for i in range(4):
            inset.scatter(i * nx + nx/2 - 0.5, nx/2 - 0.5, marker='*', c='white', s=snsize)
            for dx, dy in dxdypred_all:
                xh = i * nx + (nx - 1) / 2 + dx / 2**(4-i)
                yh = (ny - 1) / 2 + dy / 2**(4-i)
                if xh - i * nx >= 0 and xh - i * nx <= nx and yh >= -ny and yh <= 2 * ny:
                    inset.scatter(xh, yh, lw=3, marker='.', s=20 * (i + 1), color='none', edgecolor='violet')
            xh = i * nx + (nx - 1) / 2 + dxpred / 2**(4-i)
            yh = (ny - 1) / 2 + dypred / 2**(4-i)
            if xh - i * nx >= 0 and xh - i * nx <= nx and yh >= -ny and yh <= 2 * ny:
                inset.scatter(xh, yh, lw=3, marker='o', s=200, color='none', edgecolor='white')#, width=0.2, head_width=0.5, length_includes_head=True)
            inset.axvline(i * nx - 0.5, c='gray')
            xs = np.array([nx / 2 - nx / 4 - 0.5,
                           nx / 2 - nx / 4 - 0.5,
                           nx / 2 + nx / 4 - 0.5,
                           nx / 2 + nx / 4 - 0.5,
                           nx / 2 - nx / 4 - 0.5])
            ys = [nx / 2 - nx / 4 - 0.5,
                  nx / 2 + nx / 4 - 0.5,
                  nx / 2 + nx / 4 - 0.5,
                  nx / 2 - nx / 4 - 0.5,
                  nx / 2 - nx / 4 - 0.5]
            inset.plot(i * nx + xs, ys, c='k', ls=":", lw=2)
            inset.plot([i * nx + xs[1], (i + 1) * nx - 0.5], [ys[1], nx - 0.5], c='k', ls=':', lw=2)
            inset.plot([i * nx + xs[0], (i + 1) * nx - 0.5], [ys[0], - 0.5], c='k', ls=":", lw=2)

        inset.scatter(4 * nx + nx/2 - 0.5, nx/2 - 0.5, marker='*', c='white', s=snsize)
        for dx, dy in dxdypred_all:
            xh = 4 * nx + (nx - 1) / 2 + dx
            yh = (ny - 1) / 2 + dy
            if xh - 4 * nx >= 0 and xh - 4 * nx <= nx and yh >= -ny and yh <= 2 * ny:
                inset.scatter(xh, yh, lw=3, marker='.', s=100, color='none', edgecolor='violet')
        xh = 4 * nx + (nx - 1) / 2 + dxpred
        yh = (ny - 1) / 2 + dypred
        if xh - 4 * nx >= 0 and xh - 4 * nx <= nx and yh >= -ny and yh <= 2 * ny:
            inset.scatter(xh, yh, lw=3, marker='o', s=200, color='none', edgecolor='white')
        inset.spines['bottom'].set_color('gray')
        inset.spines['top'].set_color('gray')
        inset.spines['left'].set_color('gray')
        inset.spines['right'].set_color('gray')
        inset.set_xlim(-0.5, image.shape[1] - 0.5)
        inset.set_ylim(-0.5, image.shape[0] - 0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.imagefolder, "%s_comp.pdf" % oid))
        plt.savefig(os.path.join(self.imagefolder, "%s_comp.png" % oid))
        plt.show()

    def plot_size(self, image, objects=None, axis=None, doplot=True):

        """plot the host size estimation method images"""

        """
        Parameters
        ----------
        image : numpy array
           numpy array with 2D image
        objects : pandas dataframe
           dataframe with SExtractor detected sources
        axis : matplotlib axis
           axis where to plot
        doplot : bool
           whether to plot images associated to the method
        """
        
        # plot background-subtracted image
        if doplot:
            image_masked = np.ma.masked_where((image <= 0), image)
            norm = ImageNormalize(image_masked[image_masked > 0], interval=ZScaleInterval())
            im = axis.imshow(image_masked, interpolation='nearest', cmap='viridis', origin='lower', \
                        norm=norm)#, vmin=np.min(image_masked), vmax= np.percentile(image_masked, 99))
        # plot an ellipse for each object
        if objects is not None:
            
            if objects.shape[0] == 0:
                return objects
            
            if doplot and image.shape[0] <= 30:
                axis.scatter(objects.iloc[0].xhost, objects.iloc[0].yhost, s=200, color='white')
    
            objects["ndisthost"] = objects.apply(lambda row: 
                                        self.ell_dist(row.x, row.y, row.xhost, row.yhost, row.cxx, row.cxy, row.cyy), axis=1)
            objects['cx'] = objects['cy'] = image.shape[0] / 2.
            objects["ndist"] = objects.apply(lambda row: 
                                        self.ell_dist(row.x, row.y, row.cx, row.cy, row.cxx, row.cxy, row.cyy), axis=1)
            objects = objects.sort_values("ndisthost")
    
            if doplot:
                first = True
                for idx, row in objects.iterrows():
        
                    e = Ellipse(xy=(row.x - 0.5, row.y - 0.5),
                            width=6*row.a, height=6*row.b,
                            angle=row.theta * 180. / np.pi)
                    e.set_facecolor('none')
                    if first:
                        e.set_edgecolor('k')
                        e.set_linewidth(3)
                        first = False
                        xc = image.shape[0]/2
                        if xc < 30:
                            ann = axis.arrow(xc, xc, objects.iloc[0].x - xc, objects.iloc[0].y - xc, color='k', width=0.3, head_width=3, length_includes_head=True)
                        else:
                            ann = axis.arrow(240, 240, (objects.iloc[0].x - xc), (objects.iloc[0].y - xc), color='k', width=3, head_width=30, length_includes_head=True)
                        ann.set_clip_box(axis.bbox)
                        ann.set_in_layout(False)
    
                    else:
                        e.set_edgecolor('white')
                    axis.add_artist(e)
                    e.set_clip_box(axis.bbox)
                    e.set_in_layout(False)
    
                axis.set_xlim(-0.5, image.shape[1] - 0.5)
                axis.set_ylim(-0.5, image.shape[1] - 0.5)
        
        return objects        

    def get_hostsize(self, oid, doplot=False):

        """estimate the host size"""

        """
        Parameters
        ----------
        oid : String
           object identifier
        doplot : bool
           whether to plot the method associated images
        """
        
        idx_host = self.df.index.get_loc(oid)

        if ("xhost" not in self.df or "yhost" not in self.df) and ("dx_delight" not in self.df or "dy_delight" not in self.df):
            print("Host coordinates xhost, yhost not available, or prediction not availableSN2004aq")
            return
        
        xhost = self.df.loc[oid].xSN + self.df.loc[oid].dx_delight
        yhost = self.df.loc[oid].ySN + self.df.loc[oid].dy_delight

        # recompute multi resolution image without masking
        unmaskeddata = self.df.loc[[oid]].apply(lambda row: self.get_data(row.name, os.path.join(self.downloadfolder, str(row.filename)), row.dx, row.dy, self.nlevels, False, False, False), axis=1, result_type='expand')
        unmaskeddata.columns = ['data', 'mask']

        if doplot:
            fig, ax = plt.subplots(ncols=self.nlevels+1, figsize=(self.nlevels*5, 5))
        mindistsize = 1e9
        for i in np.array(range(self.nlevels+1))[::-1]:
            if doplot:
                ax[self.nlevels-i].axis(False)
            if i == self.nlevels:
                if doplot:
                    ax[self.nlevels-i].text(0, 460, "Original", c='k', fontsize=35, va='top')
                    ax[self.nlevels-i].scatter([xhost], [yhost], c='r', s=50, zorder=10000)
                image = fits.open(os.path.join(self.downloadfolder, self.df.loc[oid].filename))[0].data
                image = image.byteswap().newbyteorder()
            else:
                image = self.X[idx_host, i, :, :]
                unmaskedimage = np.stack(unmaskeddata.data.values)[0, i]
                if doplot:
                    ax[self.nlevels-i].text(0, 28, f"level {self.nlevels-i}", c='white', fontsize=35, va='top')
                    ax[self.nlevels-i].scatter([image.shape[0]/2. + (xhost - 240) / 2**i], [image.shape[0]/2. + (yhost - 240) / 2**i], c='r', s=50, zorder=10000)
                    ax[self.nlevels-i].set_ylim(-0.5, 30 - 0.5)
            #print(oid, i, image.shape)
            if i == self.nlevels:
                objects, mask = self.get_objects(image)
            else:
                objects, mask = self.get_objects(unmaskedimage)
            objects = pd.DataFrame(data = objects, columns = ['thresh', 'npix', 'tnpix', 'xmin', 'xmax', 'ymin', 'ymax', 'x', 'y', 'x2', 'y2', 'xy', 'errx2', 'erry2', 'errxy', 'a', 'b', 'theta', \
            'cxx', 'cyy', 'cxy', 'cflux', 'flux', 'cpeak', 'peak', 'xcpeak', 'ycpeak', 'xpeak', 'ypeak', 'flag'])
            objects['cx'] = objects['cy'] = image.shape[0] / 2.
            objects["xhost"] = image.shape[0] / 2. + (xhost - 240) / 2**i
            objects["yhost"] = image.shape[0] / 2. + (yhost - 240) / 2**i

            if doplot:
                if i == self.nlevels:
                    objects = self.plot_size(image, objects=objects, axis=ax[self.nlevels-i], doplot=doplot)
                else:
                    objects = self.plot_size(unmaskedimage, objects=objects, axis=ax[self.nlevels-i], doplot=doplot)
            else:
                if i == self.nlevels:
                    objects = self.plot_size(image, objects=objects, axis=None, doplot=False)
                else:
                    objects = self.plot_size(unmaskedimage, objects=objects, axis=None, doplot=False)

            if objects is not None:
                if objects.shape[0] > 0:
                    if i == self.nlevels:
                        size = objects.iloc[0][["a", "b"]].max()
                        dist = objects.iloc[0]["ndisthost"]
                        distSN = objects.iloc[0]["ndist"]
                    else:
                        size = objects.iloc[0][["a", "b"]].max() * 2**i
                        dist = objects.iloc[0]["ndisthost"] * 2**i
                        distSN = objects.iloc[0]["ndist"] * 2**i
                        #print(objects.iloc[0]["ndist"], dist)
                    if doplot:
                        ax[self.nlevels-i].set_title("$a$: %.1f\", error/$a$: %.3f" % (size * self.pixscale, dist/size), fontsize=24)
                    if i < self.nlevels and dist/size < mindistsize and mindistsize > 0.05:
                        mindistsize = dist/size
                        hostsize = size
                        hostsep = dist
                        SNsep = distSN
        if doplot:
            plt.tight_layout()
            plt.savefig(os.path.join(self.imagefolder, f"{oid}_semimajor.pdf"))
            plt.show()

        if "hostsize" in locals():
            self.df.loc[oid, "hostsize"] = hostsize * self.pixscale
            self.df.loc[oid, "hostsep"] = hostsep * self.pixscale
            self.df.loc[oid, "mindistsize"] = mindistsize
            if doplot:
                print(f"{oid} predicted host estimated semi-major axis: {self.df.loc[oid]['hostsize']} arcsec")
                if self.df.loc[oid]["mindistsize"] > 0.05:
                    print(f"   WARNING: center of SExtractor host position is further than 5% the semi-major axis from the predicted position (ratio:{self.df.loc[oid]['mindistsize']}).")
        else:
            print(f"   WARNING: Multi-resolution image of {oid} has no objects detected, cannot look for host size.")
            self.df.loc[oid, "hostsize"] = np.nan
            self.df.loc[oid, "hostsep"] = np.nan
            self.df.loc[oid, "mindistsize"] = np.nan


    def preprocess(self):

        """preprocess data"""

        """
        Parameters
        ----------
        None
        """
        
        self.oids = self.df.index.to_numpy()
        
        # normalize
        norm = np.max(self.X.reshape((self.X.shape[0], self.X.shape[1], self.X.shape[2] * self.X.shape[3])), axis=2)
        mask = np.isfinite(np.max(norm, axis=1)) & (np.max(norm, axis=1) > 0)
        if mask.shape[0] != mask.sum():
            print("   WARNING: removing %i objects with non finite or only zero values in the multi-resolution images:" % (mask.shape[0] - mask.sum()))
            print("   Please remove %s from your sample." % (", ".join(self.oids[np.invert(mask)])))
            self.oids = self.oids[mask]
            self.X = self.X[mask]
        
        # normalize wrt maximum value
        norm = np.max(self.X.reshape((self.X.shape[0], self.X.shape[1], self.X.shape[2] * self.X.shape[3])), axis=2)
        norm[norm == 0] = 1
        self.Xpr = self.X / norm[:, :, np.newaxis, np.newaxis]
        
        # delete auxiliary variables
        del(norm, mask)
        
        # swap axis
        self.Xpr = np.swapaxes(self.Xpr, 1, 3)
        if np.shape(self.oids)[0] != np.shape(self.X)[0]:
            print("WARNING", np.shape(self.oids), np.shape(self.X))
        
    
    def load_model(self, modelversion='v1', modelfile=None):

        """load tensorflow model"""

        """
        Parameters
        ----------
        modelversion : string
           optional suffix of tensorflow model, default is v1
        modelfile : string
           optional filename, use custom model file, it has priority over the previous modelversion string
        """

        if modelfile is None:
            self.modelfile = pkg_resources.resource_filename(__name__, f'DELIGHT_{modelversion}.h5')
        else:
            self.modelfile = modelfile

        self.tfmodel = tf.keras.models.load_model(self.modelfile)
    

    def derotate(self, y_pred, reg=False):

        """derotate output"""

        """
        Parameters
        ----------
        data : numpy array
           numpy array with 2D image
        """

        
        if not reg:
            return np.dstack([y_pred.reshape((y_pred.shape[0], 8, 2))[:, 0],
              y_pred.reshape((y_pred.shape[0], 8, 2))[:, 1, ::-1] * [1, -1],
              y_pred.reshape((y_pred.shape[0], 8, 2))[:, 2, :] * [-1, -1],
              y_pred.reshape((y_pred.shape[0], 8, 2))[:, 3, ::-1] * [-1, 1],
              y_pred.reshape((y_pred.shape[0], 8, 2))[:, 4, :] * [1, -1],
              y_pred.reshape((y_pred.shape[0], 8, 2))[:, 5, ::-1],
              y_pred.reshape((y_pred.shape[0], 8, 2))[:, 6, :] * [-1, 1],
              y_pred.reshape((y_pred.shape[0], 8, 2))[:, 7, ::-1] * [-1, -1]]).reshape((y_pred.shape[0], 2, 8)).swapaxes(1, 2)
        else:
            return np.dstack([y_pred.reshape((y_pred.shape[0], 8, 3))[:, 0],
              y_pred.reshape((y_pred.shape[0], 8, 3))[:, 1, ::-1] * [1, -1, 1],
              y_pred.reshape((y_pred.shape[0], 8, 3))[:, 2, :] * [-1, -1, 1],
              y_pred.reshape((y_pred.shape[0], 8, 3))[:, 3, ::-1] * [-1, 1, 1],
              y_pred.reshape((y_pred.shape[0], 8, 3))[:, 4, :] * [1, -1, 1],
              y_pred.reshape((y_pred.shape[0], 8, 3))[:, 5, ::-1],
              y_pred.reshape((y_pred.shape[0], 8, 3))[:, 6, :] * [-1, 1, 1],
              y_pred.reshape((y_pred.shape[0], 8, 3))[:, 7, ::-1] * [-1, -1, 1]]).reshape((y_pred.shape[0], 3, 8)).swapaxes(1, 2)
        
        return y_pred


    def predict(self):

        """predict positions"""

        """
        Parameters
        ----------
        data : numpy array
           numpy array with 2D image
        """

        
        y_pred = self.tfmodel.predict([self.Xpr[:, :, :, i] for i in range(self.Xpr.shape[3])])

        y_pred = self.derotate(y_pred)

        # mean prediction
        y_pred_mean = y_pred.mean(axis=1)
        # root mean squared differences
        y_pred_std = np.sqrt((((y_pred - y_pred_mean[:, np.newaxis, :])**2).sum(axis=2)).mean(axis=1))

        
        self.df["dxdy_delight_rotflip"] = [ys for ys in y_pred] 
        self.df["dx_delight"] = y_pred_mean[:, 0]
        self.df["dy_delight"] = y_pred_mean[:, 1]
        self.df["std_delight"] = y_pred_std

        self.df["host_coords_delight_pred"] = self.df.apply(lambda row: row.wcs.pixel_to_world(row.xSN + row.dx_delight, row.ySN + row.dy_delight), axis=1)
        self.df["host_coords_sex_pred"] = self.df.apply(lambda row: row.wcs.pixel_to_world(row.xSN + row.dx_sex, row.ySN + row.dy_sex), axis=1)


        self.df["ra_delight"] = self.df["host_coords_delight_pred"].apply(lambda row: float(row.ra / u.deg))
        self.df["dec_delight"] = self.df["host_coords_delight_pred"].apply(lambda row: float(row.dec / u.deg))
        self.df["ra_sex"] = self.df["host_coords_sex_pred"].apply(lambda row: float(row.ra / u.deg))
        self.df["dec_sex"] = self.df["host_coords_sex_pred"].apply(lambda row: float(row.dec / u.deg))
