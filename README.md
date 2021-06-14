# Installation
## Conda
Install the 3d_vision toolbox with
```
conda develop $HOME/3d_vision/vision_3d_utils
```

## Minimum required dataset
The following files are required to compute the projection, the occlusion detection, and the reflectance detection:
```
|_detail
|--*
|images
|--scene_cam_CC_final_hdf5
|----frame.FFFF.color.hdf5
|----frame.FFFF.diffuse_reflectance.hdf5
|--scene_cam_CC_geometry.hdf5
|----frame.FFFF.position.hdf5
```

The minimum dataset is still 500 Gb, thus we have to download it through the official source.
Use [99991's alternative downloader](https://github.com/apple/ml-hypersim/tree/master/contrib/99991) to download the reduced dataset:

HDF5
```
python3 ./download.py -d /diretory/to/download/to --contains _detail --silent
python3 ./download.py -d /diretory/to/download/to --contains .color.hdf5 --silent
python3 ./download.py -d /diretory/to/download/to --contains .diffuse_reflectance.hdf5 --silent
python3 ./download.py -d /diretory/to/download/to --contains .position.hdf5 --silent
```

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

## Dependencies
```
pip3 install opencv-python h5py pandas numpy
```
