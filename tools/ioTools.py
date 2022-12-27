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

def saveCsvTrack(basedir, data, columns):
    data = pd.DataFrame(data, columns = columns)
    
    mkdirs(basedir)
    
    existingResults = glob(basedir + r'\manualResults*.csv')
    
    resultID = 0
    if len(existingResults) > 0:
        for i in range(len(existingResults)):
            resultID = np.max((resultID, int(re.search('manualResults(.+?).csv', existingResults[i]).group(1))+1))

    data.to_csv(basedir + r"\manualResults{0:03}.csv".format(resultID))