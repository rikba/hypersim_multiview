# Installation
## Conda
Install the 3d_vision toolbox with
```
conda develop $HOME/3d_vision/tools
```

# projection.py
This tools implements projection of image pixels from a camera source frame into a camera target frame.
The user can click points in the source frame and they appear in the target frame.
![image](https://user-images.githubusercontent.com/11293852/114523636-c0208d00-9c44-11eb-8cb6-13a9bce4aa20.png)
## How to use
```
python3 projection.py --help
usage: projection.py [-h]
                     data_folder volume_number scene_number cam_trajectory
                     source_frame target_frame [occlusion_threshold]
                     [reflectance_threshold]

Manually verify transformation between two hypersim scenes.

positional arguments:
  data_folder           The folder containing the hypersim scenes.
  volume_number         The volume number.
  scene_number          The scene number.
  cam_trajectory        The camera trajectory number.
  source_frame          The source frame.
  target_frame          The target frame.
  occlusion_threshold   The distance between source point and target point to
                        detect occlusions in [m].
  reflectance_threshold
                        An adaptive threshold [0..1] to detect reflections.
```
Example:
```
python3 projection.py /home/rik/data/hypersim 1 1 0 0 3 0.03 0.1
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
```
python3 ./download.py -d /diretory/to/download/to --contains _detail --silent
python3 ./download.py -d /diretory/to/download/to --contains .color.hdf5 --silent
python3 ./download.py -d /diretory/to/download/to --contains .diffuse_reflectance.hdf5 --silent
python3 ./download.py -d /diretory/to/download/to --contains .position.hdf5 --silent
```
The download script will first list all files that it is skipping.
This may take 15 minutes.
Then it starts downloading.

## Dependencies
```
pip3 install opencv-python h5py pandas numpy
```
