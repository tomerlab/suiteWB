# suiteWB
 Suite for whole brain TH mapping

# Pipeline
## Registration
Create downsampled whole brain nrrd file with resolution 10x10x10 $\mu m$. Register the brain volumes onto the average reference atlas. These transformation parameters are applied to the high resolution brain nrrd image. 

## Segmentation
### cropRegions.ipynb

Cropped the high density/low density ROI from the original image for separate ilastik processing. 

### ilastik processing 

### denseRegionSegmentation.ipynb / sparseRegionSegmentation.ipynb

After ilastik processing and generating probability map, these steps use difference of Gaussians (DoG) blob detection to extract cell locations. Optional masks can be used to further divide the region into subsets where different thresholds can be used for blob detection at different locations.

### manualProofread.ipynb

Load the segmented data and inspect through Napari. Point cloud can be edited by manually deleting and adding points.

### mergeBlobs.ipynb

Merge the blobs detected from sparse and dense regions back to original image coordinates.

### assignAtlas.ipynb

Assign the Allen Brain Atlas hierarchy tree regions to each cell. A csv of the cell-region mapping is generated.
