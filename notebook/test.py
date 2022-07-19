import os, sys
import pandas as pd

sys.path.append("../lib")
from TransientSet import *

# data directory
datadir = '../data'
df = pd.read_csv(os.path.join(datadir, 'testcoords.csv'))

# start TransientSet object
ts = TransientSet(datadir, df.oid.values, df.ra.values, df.dec.values)
ts.download()    

ts.getpixcoords()

nlevels = 5
domask = False
doobject = True

doplot = True
ts.compute_multiresolution(nlevels, domask, doobject, doplot)

print(ts.df)
