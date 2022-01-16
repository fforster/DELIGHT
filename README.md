Delight

model

parameters

methods:

train_model(train_transients, test_transients)

save_model()

load_model()

eval_model(transients)

Transients

filedir

coordinates # ra, dec

peakmags # peak magnitudes

redshifts # redshifts

host_images # list of host images

host_multi_level_images # list of host multi resolution images

methods:

init(filedir, coordinates, [peakmags, redshifts]) # initialization

download() # download missing data

load_images([nlevels, domask, dosex]) # load original or resamples images

resample(nlevels, domask, dosex) # resample data to nlevels

save_images() # save resampled data

HostImage

filename # original filename

pixelsize # pixel size

data # data

methods:

resample(nlevels, domask, dosex)

HostMultiLevelImage

filename # filename with new resolution

nlevels # number of levels

dimensions # number of pixels per level

domask # whether to use mask

dosex # whether to use sextractor

data # multi resolution data
