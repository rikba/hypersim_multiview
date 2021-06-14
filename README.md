# Hypersim Multi View
This repository implements frame transformation methods to interface and use the [Apple Hypersim Evermotion data set](https://github.com/apple/ml-hypersim) with multi view geometry applications.
For example this package can be used to label pixel correspondences for self-supervised key point detection and description as in [KP2D](https://github.com/TRI-ML/KP2D)

## Frame-to-frame pixel projection
Pixel correspondences between two frames of the same scene can be found through the pixel world position in the source frame and the target camera pose and camera intrinsic calibration.

`warp()`:
<p float="left">
  <img src="https://user-images.githubusercontent.com/11293852/121869932-d4474000-cd02-11eb-9ec2-d773ee1cbae2.png" alt="Frame-to-frame pixel projection, source" height="240"/>
  <img src="https://user-images.githubusercontent.com/11293852/121869938-d6110380-cd02-11eb-8644-2ef423608e06.png" alt="Frame-to-frame pixel projection, target" height="240"/>
</p>

## Occlusion detection
Pixel that are visible in the source frame but occluded in the target frame can be detected automatically.
<img src="https://user-images.githubusercontent.com/11293852/121869958-db6e4e00-cd02-11eb-8ebc-b83d669fc641.png" alt="Occlusion detection" height="240"/>

## Reflectance detection
Reflecting and transparent surface can be masked using the diffuse reflectance information of the Hypersim dataset.
<img src="https://user-images.githubusercontent.com/11293852/121869955-da3d2100-cd02-11eb-9915-7f047708ecb9.png" alt="Reflectance detection" height="240"/>


# Installation
Install the hypersim_multiview toolbox with
```
conda develop $HOME/hypersim_multiview
```
or
```
pip3 install -e $HOME/hypersim_multiview
```

# Minimum required dataset
The following files are required to compute the projection, the occlusion detection, and the reflectance detection:
```
|_detail
|--*
|images
|--scene_cam_CC_final_preview
|----frame.FFFF.color.jpg
|----frame.FFFF.diffuse_reflectance.jpg
|--scene_cam_CC_geometry.hdf5
|----frame.FFFF.position.hdf5
```

The minimum dataset is still 500 Gb, thus we have to download it through the official source.
Use [99991's alternative downloader](https://github.com/apple/ml-hypersim/tree/b125e8fa4f55539cbb2237ddb052504bf7d377bc/contrib/99991) to download the reduced dataset:

JPG
```
python3 ./download.py -d /diretory/to/download/to --contains _detail --silent
python3 ./download.py -d /diretory/to/download/to --contains .color.jpg --contains final_preview --silent
python3 ./download.py -d /diretory/to/download/to --contains .diffuse_reflectance.jpg --silent
python3 ./download.py -d /diretory/to/download/to --contains .position.hdf5 --silent
```

The download script will first list all files that it is skipping.
This may take 15 minutes.
Then it starts downloading.
