# suiteWB
 Tools for whole brain phenotyping

# Pipeline
## Registration
Create downsampled whole brain nrrd file with resolution 10x10x10 $\mu m$. Register the brain volumes onto the average reference atlas. These transformation parameters are applied to the high resolution brain nrrd image. 

## Segmentation
### cropRegions.ipynb

Crops the high density/low density ROI from the original image for independent ilastik processing. 

### ilastik processing 

### denseRegionSegmentation.ipynb / sparseRegionSegmentation.ipynb

After ilastik processing and probability map generation, difference of Gaussians (DoG) method is used to extract cells and their coordinates. Optional masks can be used to divide the region into subsets where different thresholds can be used for blob detection at different locations.

### manualProofread.ipynb

Loads the segmented data and inspect through Napari. Point cloud can be edited by manually deleting and adding points.

### mergeBlobs.ipynb

Merges the blobs detected from sparse and dense regions back to original image coordinates.

### assignAtlas.ipynb

Assign the Allen Brain Atlas hierarchy tree regions to each detected cell. A csv of the cell-region mapping is generated.
