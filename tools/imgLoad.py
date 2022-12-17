import numpy as np
from glob import glob

from skimage.io import imread
from skimage.io.collection import alphanumeric_key
from dask import delayed
import dask.array as da
from PIL import Image
import tifffile

import warnings
warnings.filterwarnings('ignore')



def loadVirtualTifSeq(pathname, order = 'xyz', isPrint = True):
    tifNames = sorted(glob(pathname, recursive = True), key=alphanumeric_key)

    sample = imread(tifNames[0])

    lazy_imread = delayed(imread)  # lazy reader
    lazy_arrays = [lazy_imread(fn) for fn in tifNames]
    dask_arrays = [
        da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
        for delayed_reader in lazy_arrays
    ]
    
    if order == 'xyz':
        img = da.stack(dask_arrays, axis = 2)
    elif order == 'zxy':
        img = da.stack(dask_arrays, axis = 0)
    
    if isPrint:
        print(img.shape)  
    
    return img