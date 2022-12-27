import numpy as np
import skimage.feature as skfeature
import pandas as pd
import os
import matplotlib.pyplot as plt
from glob import glob
import napari


import sys
sys.path.append(r'../')
from tools import ioTools as tio


class Block(object):
    def __init__(self, starting, ending):
        self.x1, self.y1, self.z1 = starting
        self.x2, self.y2, self.z2 = ending
        
class Blocks(object):
    def __init__(self, imgShape, blockShape, blockOverlap):
        self.imgShape = np.array(imgShape)
        self.blockShape = np.array(blockShape)
        self.blockOverlap = np.array(blockOverlap)

        self.blockNumber = np.ceil((self.imgShape - self.blockOverlap)  \
                                   /(self.blockShape - self.blockOverlap)).astype('uint16')
        print('Block Number (x,y,z): ', self.blockNumber)
        
        self.blocks = [[[None for _ in range(self.blockNumber[2])] \
                        for _ in range(self.blockNumber[1])] \
                       for _ in range(self.blockNumber[0])]
        
        for x in range(self.blockNumber[0]):
            for y in range(self.blockNumber[1]):
                for z in range(self.blockNumber[2]):
                    starting = (self.blockShape - self.blockOverlap) * np.array((x,y,z))
                    ending = np.min(np.array((starting + self.blockShape,self.imgShape)), axis = 0)

                    self.blocks[x][y][z] = Block(starting, ending)
                    
    def visualize(self, patchImgGap = 100):                
        patchImg = np.zeros((self.imgShape[0]+self.imgShape[2]+patchImgGap, \
                             self.imgShape[1]+self.imgShape[2]+patchImgGap))

        for x in range(self.blockNumber[0]):
            for y in range(self.blockNumber[1]):
                patchImg[self.blocks[x][y][0].x1 : self.blocks[x][y][0].x2, \
                         self.blocks[x][y][0].y1 : self.blocks[x][y][0].y2] += 1
                
                for z in range(self.blockNumber[2]):
                    
                    if x == 0:
                        xOffset = patchImgGap + self.imgShape[0]
                        yOffset = 0
                        patchImg[self.blocks[x][y][z].z1 + xOffset : self.blocks[x][y][z].z2 + xOffset, \
                                 self.blocks[x][y][z].y1 + yOffset : self.blocks[x][y][z].y2 + yOffset] += 1
                    if y == 0:
                        xOffset = 0
                        yOffset = patchImgGap + self.imgShape[1]
                        patchImg[self.blocks[x][y][z].x1 + xOffset : self.blocks[x][y][z].x2 + xOffset, \
                                 self.blocks[x][y][z].z1 + yOffset : self.blocks[x][y][z].z2 + yOffset] += 1

                
                
        plt.figure(figsize = (8,10))
        plt.imshow(patchImg)
        
    def crop_block(self, img, blockId):
        x, y, z = blockId
        block = self.blocks[x][y][z]
        im_block = np.array(img[block.x1:block.x2,
                     block.y1:block.y2,
                     block.z1:block.z2])
        return im_block
    
    
class Segmentation(object):
    def __init__(self, img, prob, masks, blocks):
        self.img = img
        self.prob = prob
        self.masks = masks
        self.blocks = blocks
        self.blockId = [-1,-1,-1]
        
    def segmentBlock(self, blockId, thresholds, onlyMask = True, min_sigma = [2, 2, 1], max_sigma = [4, 4, 4], sigma_ratio = 1.6, \
                     probThresh = 60):
        
        self.checkParam(thresholds, onlyMask)
        
        print('Load data...')
        if self.blockId != blockId:
            self.blockId = blockId
            self.blockProb = self.blocks.crop_block(self.prob, self.blockId)
            self.blockImg = self.blocks.crop_block(self.img, self.blockId)
            self.blockMasks = [self.blocks.crop_block(self.masks[i], self.blockId) for i in range(len(self.masks))]
        nMasks = len(self.blockMasks)
        
        labeledMask = np.zeros_like(self.blockMasks[0])
        for i in range(nMasks-1, -1, -1):
            labeledMask[self.blockMasks[i] > 0] =  i + 1
        labeledMask = labeledMask.astype('uint8')    
        
        if not onlyMask:
            labeledMask += 1
            nMasks += 1
        
        blockBlobs = np.zeros((0,7))
        for iMask in range(nMasks):
            print('Blob detection (Mask {})...'.format(iMask+1))
            if onlyMask and ((labeledMask == iMask + 1).sum() == 0):
                print('Detection: 0')
                continue
            
            blobs = skfeature.blob_dog(self.blockProb, min_sigma=min_sigma, max_sigma=max_sigma, \
                                       sigma_ratio=sigma_ratio, threshold=thresholds[iMask], overlap=0.2)
            if len(blobs) > 0:
                blobs = blobs.astype('uint16')
                selectID = []
                for iBlob, blob in enumerate(blobs):
                    if self.blockProb[blob[0], blob[1], blob[2]] > probThresh and \
                       labeledMask[blob[0], blob[1], blob[2]] == iMask + 1:
                        selectID += [iBlob]
                print('Detection: {}'.format(len(selectID)))
                blobsResult = np.concatenate((blobs[selectID, :], (iMask+1) * np.ones((len(selectID), 1))), axis = 1)
            else:
                blobsResult = np.zeros((0,7))
            blockBlobs = np.concatenate((blockBlobs, blobsResult), axis = 0)
        
        return  pd.DataFrame(blockBlobs, columns = ['x', 'y', 'z', 'rx','ry','rz', 'mask'])
    
    def segmentWhole(self, saveDir, thresholds, onlyMask = True, min_sigma = [2, 2, 1], max_sigma = [4, 4, 4], sigma_ratio = 1.6, \
                     probThresh = 60, subdir = r'', paramFile = r'param'):
        
        self.checkParam(thresholds, onlyMask)
        
        tempLoc = saveDir + subdir + r'\blocks'
        tio.mkdirs(tempLoc)

        tio.saveNpy(saveDir + r'\parameters\\' + paramFile + r'.npy', thresholds = thresholds,  min_sigma=min_sigma, max_sigma=max_sigma, \
                                       sigma_ratio=sigma_ratio, probThresh = probThresh, onlyMask = onlyMask)
        
        self.allBlobs = pd.DataFrame({})
        for bz in range(self.blocks.blockNumber[2]):
            if bz > 0:
                thresholds = [0.01, 0.04]
            if bz > 1:
                thresholds = [0.02, 0.2]

            for by in range(self.blocks.blockNumber[1]):
                for bx in range(self.blocks.blockNumber[0]):
                    print('\n===')
                    print('Block: ', bx, by, bz)
                    
                    blockBlobs = self.segmentBlock([bx, by, bz], thresholds, onlyMask)
                    
                    if len(blockBlobs) > 0:
                        blockBlobs = self.filterBoundaryCells(blockBlobs)
                        blockBlobs = self.mapFullCoord(blockBlobs, bx, by, bz)
                        self.allBlobs = self.allBlobs.append(blockBlobs)
                    blockBlobs.to_csv(tempLoc + r'\{0:02d}_{1:02d}_{2:02d}.csv'.format(bx,by,bz))

                        
        return self.allBlobs
    
    def loadBlockResults(self, loadDir):

        self.allBlobs = pd.DataFrame({})
        for bz in range(self.blocks.blockNumber[2]):
            for by in range(self.blocks.blockNumber[1]):
                for bx in range(self.blocks.blockNumber[0]):
                    try:
                        blockBlobs = pd.read_csv(loadDir + r'\{0:02d}_{1:02d}_{2:02d}.csv'.format(bx,by,bz))
                        self.allBlobs = self.allBlobs.append(blockBlobs)
                    except:
                        pass
        return self.allBlobs
                        
    def filterBoundaryCells(self, blobs):
        blockOverlap = self.blocks.blockOverlap
        blockShape = self.blocks.blockShape
        selectId = (blobs.x >= blockOverlap[0]/2) & (blobs.x < blockShape[0] - blockOverlap[0]/2) & \
            (blobs.y >= blockOverlap[1]/2) & (blobs.y < blockShape[1] - blockOverlap[1]/2) & \
            (blobs.z >= blockOverlap[2]/2) & (blobs.z < blockShape[2] - blockOverlap[2]/2)

        return blobs[selectId]
    
    def mapFullCoord(self, blobs, bx, by, bz):
        blobs.x += self.blocks.blocks[bx][by][bz].x1
        blobs.y += self.blocks.blocks[bx][by][bz].y1
        blobs.z += self.blocks.blocks[bx][by][bz].z1

        return blobs
    
    def checkParam(self, thresholds, onlyMask):
        nMasks = len(self.masks) + 1 - int(onlyMask)
        nThresholds = len(thresholds)
        assert nThresholds == nMasks, f"Masks number ({nMasks}) unequal to thresholds number ({nThresholds})"
        
        
class Proofread(object):
    def __init__(self, basedir, img, tempFolder = r'\proofreadTemp'):
        self.basedir = basedir
        self.tempFolder = tempFolder
        self.cells = self.loadResults()
        self.img = img
        
    def loadResults(self):
        assert len(glob(self.basedir + r'\results.csv')) > 0, f'Please generate results first'

        existingResults = glob(self.basedir + self.tempFolder + r'\manualResults*.csv')
            
        if len(existingResults) > 0:
            self.latestID = 0
            for i in range(len(existingResults)):
                self.latestID = np.max((self.latestID, int(re.search('manualResults(.+?).csv', existingResults[i]).group(1))))
            cells = pd.read_csv(self.basedir + self.tempFolder + r'\manualResults{0:03}.csv'.format(self.latestID))
        else:
            cells = pd.read_csv(self.basedir + r'\results.csv')
            
        return cells

    def viewResults(self):

        self.viewer = napari.view_image(self.img, contrast_limits=[0,10000], colormap = 'gray',  scale = [2,1,1], name = 'Img')

        self.viewer.add_points(np.array([self.cells.z, self.cells.x, self.cells.y]).T, size = 6, scale = [2,1,1], 
                          n_dimensional = True, face_color = 'red', name = 'Cells')
        
    def updateResults(self):
        self.cells = pd.DataFrame(self.viewer.layers[1].data, columns = ['z', 'x','y'])
        
        tio.mkdirs(self.basedir + self.tempFolder )
        
        self.latestID += 1
        self.cells.to_csv(self.basedir + self.tempFolder + r'manualResults{0:03}.csv'.format(self.latestID))
        
#     def autoBackupThread(self):
#         while(1)
