import os
import numpy as np

def mkdirs(folderPath):
    try:
        os.makedirs(folderPath)
        print('Create ', folderPath)
    except:
        print('Exists ', folderPath)
        
def saveNpy(filePath, **kwargs):
    mkdirs(os.path.dirname(filePath))
    np.save(filePath, kwargs, allow_pickle = True)
    
   
def loadNpy(filePath):
    np.load(filePath, allow_pickle = True)