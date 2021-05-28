import os
import h5py
import numpy as np
import pandas as pd
import cv2

class DataLoader:
    def __init__(self, data_folder, image_type='jpg', verbose = False):
        self.data_folder = data_folder
        self.image_type = image_type # 'jpg' or 'hdf5'
        self.verbose = verbose
        self.meters_per_asset_unit = {} # Cache asset scale.
        self.t_CW = {} # Cache all cam positions of scene.
        self.R_CW = {} # Cache all cam orientations of scene.

    def getSceneFolder(self, volume, scene):
        return os.path.join(self.data_folder, 'ai_{:03d}_{:03d}'.format(int(volume), int(scene)))

    def getCamPosFolder(self, volume, scene, cam):
        return os.path.join(self.getSceneFolder(volume, scene), '_detail/cam_{:02d}'.format(int(cam)))

    def readH5(self, file):
        if self.verbose:
            print('Opening file: {:s}'.format(file))
        h5 = h5py.File(file, 'r')
        data = np.array(h5['dataset'][:], dtype=h5['dataset'].dtype)
        h5.close()

        return data

    def readJpg(self, file):
        if self.verbose:
            print('Opening file: {:s}'.format(file))
        data = cv2.imread(cv2.samples.findFile(file))

        return data

    def loadBgr(self, volume, scene, cam, frame):
        base = self.getSceneFolder(volume, scene)
        relative = 'images/scene_cam_{:02d}_final_{image_type}/frame.{:04d}.color.{image_type}'.format(int(cam), int(frame), image_type=self.image_type)
        relative = relative.replace('final_jpg', 'final_preview')
        f = os.path.join(base, relative)
        if self.image_type == 'hdf5':
            img = self.readH5(f)
        else:
            img = self.readJpg(f)
        return img

    def loadRgb(self, volume, scene, cam, frame):
        img = self.loadBgr(volume, scene, cam, frame)
        # Fix color ordering
        img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB)
        return img

    def loadReflectance(self, volume, scene, cam, frame):
        base = self.getSceneFolder(volume, scene)
        relative = 'images/scene_cam_{:02d}_final_{image_type}/frame.{:04d}.diffuse_reflectance.{image_type}'.format(int(cam), int(frame), image_type=self.image_type)
        relative = relative.replace('final_jpg', 'final_preview')
        f = os.path.join(base, relative)
        if self.image_type == 'hdf5':
            img = self.readH5(f)
        else:
            img = self.readJpg(f)
        # Fix color ordering
        img = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2BGR)
        return img

    def getAssetScale(self, volume, scene):
        key = self.getSceneFolder(volume, scene)
        if key not in self.meters_per_asset_unit:
            f = os.path.join(key, '_detail/metadata_scene.csv')
            df_scene = pd.read_csv(f, index_col='parameter_name', float_precision='round_trip')
            self.meters_per_asset_unit[key] = df_scene.loc['meters_per_asset_unit'][0]
        return self.meters_per_asset_unit[key]

    def loadPositionMap(self, volume, scene, cam, frame):
        base = self.getSceneFolder(volume, scene)
        relative = 'images/scene_cam_{:02d}_geometry_hdf5/frame.{:04d}.position.hdf5'.format(int(cam), int(frame))
        f = os.path.join(base, relative)
        img = self.readH5(f)
        # Scale positions to meters.
        img = img * self.getAssetScale(volume, scene)

        return img

    # Read and return all R_CW in 100x3x3 and t_CW in 100x3x1 camera poses of one cam trajectory.
    def loadCamPoses(self, volume, scene, cam):
        key = self.getCamPosFolder(volume, scene, cam)
        if key not in self.R_CW:
            orientation = 'camera_keyframe_orientations.hdf5'
            rot = self.readH5(os.path.join(key, orientation))
            self.R_CW[key] = np.zeros((100,3,3))
            for n, R_WC in enumerate(rot):
                self.R_CW[key][n] = R_WC.T

        if key not in self.t_CW:
            position = 'camera_keyframe_positions.hdf5'
            trans = self.readH5(os.path.join(key, position)) * self.getAssetScale(volume, scene)
            self.t_CW[key] = np.zeros((100,3))
            for n, t_WC in enumerate(trans):
                self.t_CW[key][n] = -self.R_CW[key][n].dot(t_WC)

        return self.R_CW[key], self.t_CW[key]

    # Read and return R_CW in 3x3 and t_CW in 3x1.
    def loadCamPose(self, volume, scene, cam, frame):
        R_CW, t_CW = self.loadCamPoses(volume, scene, cam)

        return R_CW[int(frame)], t_CW[int(frame)]
