import numpy as np
from glob import glob
import os
import osTool as tos
from skimage.io import imread
from skimage.io.collection import alphanumeric_key
from dask import delayed
import dask.array as da
from PIL import Image
import tifffile


def saveTifSeq(img, pathname, filename, order = 'xyz'):
    if order == 'xyz':
        pass
    elif order == 'zxy':
        img = np.transpose(img, (1,2,0))
    else:
        raise Exception('Unsupported axis order currently.')
        
    tos.mkdirs(pathname)    
    for z in range(img.shape[2]):
        im = Image.fromarray(img[:,:,z])
        im.save(pathname + r"\\" + filename +"_{0:06d}.tif".format(z))
        
    return

def saveTifSeqRGB(img, pathname, filename = 'Img', order = 'xyzc'):
    
            
    if order == 'xyzc':        
        pass
    elif order == 'cxyz':        
        img = np.transpose(img, (1,2,3,0))
    elif order == 'zxyc':   
        img = np.transpose(img, (1,2,0,3))
    elif order == 'czxy':        
        img = np.transpose(img, (2,3,1,0))
    else:
        raise Exception('Unsupported axis order currently.')
        
    tos.mkdirs(pathname)    
    for z in range(img.shape[2]):
        tifffile.imsave(pathname + r"\\" + filename +"_{0:06d}.tif".format(z), \
                        np.squeeze(img[:,:,z,:]), 'uint8')
          
        
    return
