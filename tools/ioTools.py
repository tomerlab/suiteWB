import os
import numpy as np

def mkdirs(folderPath):
    try:
        os.makedirs(folderPath)
        print('Create ', folderPath)
    except:
        print('Exists ', folderPath)
    return folderPath
        
def saveNpy(filePath, **kwargs):
    mkdirs(os.path.dirname(filePath))
    np.save(filePath, kwargs, allow_pickle = True)
    
    return filePath
   
def loadNpy(filePath):
    data = np.load(filePath, allow_pickle = True)
    
    return data