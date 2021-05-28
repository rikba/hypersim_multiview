import argparse
import numpy as np
import cv2
import os
import pandas as pd
from vision_utils.data_loader import DataLoader
from vision_utils.reprojection import Reprojection

def getWorldCoordinates(position_image, pixel):
    return position_image[int(pixel[1]), int(pixel[0])]

def detectOcclusion(source_position, source_pixel, target_position, target_pixel, occlusion_threshold):
    W_p_source = getWorldCoordinates(source_position, source_pixel)
    W_p_target = getWorldCoordinates(target_position, target_pixel)
    distance = np.linalg.norm(W_p_target - W_p_source)
    return distance > occlusion_threshold, distance

def detectReflectance(reflectance, pixel, reflectance_threshold):
    value = reflectance[int(pixel[1]), int(pixel[0])]
    return np.all(value <= reflectance_threshold), value

def detectFieldOfView(target_pixel, width, height):
    return (target_pixel[0] >= 0) and (target_pixel[0] < width) and (target_pixel[1] >= 0) and (target_pixel[1] < height)

def click_event(event, x, y, flags, params):

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # Indicate position clicked in source image.
        print('Clicked pixel: %d %d' % (x, y))
        px_source = [x, y]

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        # position in world frame
        W_p = getWorldCoordinates(img_source_position, px_source)
        print('W_p [{:.02f},{:.02f},{:.02f}]'.format(W_p[0], W_p[1], W_p[2]))

        is_reflectant, source_reflectance = detectReflectance(img_source_ref, px_source, reflectance_threshold)

        color = (255, 0, 0)
        if is_reflectant:
            print('Source pixel is reflectant [{:.02f},{:.02f},{:.02f}]!'.format(source_reflectance[0], source_reflectance[1], source_reflectance[2]))
            color = (0, 0, 255)
        cv2.putText(img_source, '[{:.02f},{:.02f},{:.02f}]'.format(W_p[0], W_p[1], W_p[2]), (x + 10, y), font,
                    0.5, color)
        cv2.circle(img_source, (x, y), 5, color)

        # Project point in world coordinates into target frame image
        p_target = reprojection.projectPixelFromSourceToTargetInt(px_source, img_source_position, R_CW, t_CW)
        #p_target = transform_point_screen_from_world(W_p)[0]
        print('Pixel in target frame: %d %d' % (p_target[0], p_target[1]))
        if reprojection.isFieldOfView(p_target):
            # Check occulusion
            is_occluded, distance = detectOcclusion(img_source_position, [x,y], img_target_position, p_target, occlusion_threshold)
            # For verification look up world position.
            W_p_target = getWorldCoordinates(img_target_position, p_target)
            print("Computed distance of source and target point: %.3f" % (distance))
            color = (255, 0, 0)
            if is_occluded:
                print("Clicked point occluded!")
                color = (0, 0, 255)

            # Check reflectance
            is_reflectant, target_reflectance = detectReflectance(img_target_ref, p_target, reflectance_threshold)
            if is_reflectant:
                print('Target pixel is reflectant [{:.02f},{:.02f},{:.02f}]!'.format(target_reflectance[0], target_reflectance[1], target_reflectance[2]))
                color = (0, 0, 255)

            # Draw projected feature.
            cv2.circle(img_target, (p_target[0], p_target[1]), 5, color)
            cv2.putText(img_target, '[{:.02f},{:.02f},{:.02f}]'.format(W_p_target[0], W_p_target[1], W_p_target[2]), (p_target[0] + 10, p_target[1]), font,
                        0.5, color)

        else:
            print("Clicked point out of bounds in target frame.")

        # Update views
        cv2.imshow('image_target', img_target)
        cv2.imshow('image_source', img_source)

# Open all images of a scene in a MxNxHxWx3 array, where M is the number of camera trajectories, N=100 is the number of images in the trajectory, and H and W are image height and width and 3 is the number of color channels.
# Additionally, opens map from image coordinate to world coordinates MxNxHxWx3.
# Returns array of all images and valid flag list, if image actually exists
def loadImages(data_folder):
    # Get number of cams
    df_cams = pd.read_csv(data_folder + '_detail/metadata_cameras.csv')
    df_scene = pd.read_csv(data_folder + '_detail/metadata_scene.csv',
                           index_col='parameter_name', float_precision='round_trip')
    meters_per_asset_unit = df_scene.loc['meters_per_asset_unit'][0]

    M = len(df_cams['camera_name'])
    N = 100
    H = 768
    W = 1024
    img = np.zeros((M,N,H,W,3))
    ref = np.zeros((M,N,H,W,3))
    pos = np.zeros((M,N,H,W,3))
    valid = np.zeros((M,N,1),dtype=bool)
    print('Loading %d camera trajectories with %d images each.' % (M,N))
    for m, cam in enumerate(df_cams['camera_name']):
        for n in range(0, N-1):
            img[m,n], ref[m,n], pos[m,n], valid[m,n] = loadImage(data_folder, m, n, meters_per_asset_unit)
    return img, ref, pos, valid

parser = argparse.ArgumentParser(
    description='Manually verify transformation between two hypersim scenes.')
parser.add_argument('data_folder', nargs=1,
                    help='The folder containing the hypersim scenes.')
parser.add_argument('volume_number', nargs=1, help='The volume number.')
parser.add_argument('scene_number', nargs=1, help='The scene number.')
parser.add_argument('cam_trajectory', nargs=1,
                    help='The camera trajectory number.')
parser.add_argument('source_frame', nargs=1, help='The source frame.')
parser.add_argument('target_frame', nargs=1, help='The target frame.')
parser.add_argument('occlusion_threshold', nargs='?', default=0.03, type=float, help='The distance between source point and target point to detect occlusions in [m].')
parser.add_argument('reflectance_threshold', nargs='?', default=30, type=int, help='An adaptive threshold [0..1] to detect reflections.')
args = parser.parse_args()

vvv = '{:03d}'.format(int(args.volume_number[0]))
nnn = '{:03d}'.format(int(args.scene_number[0]))
cam = '{:02d}'.format(int(args.cam_trajectory[0]))
s = '{:04d}'.format(int(args.source_frame[0]))
t = '{:04d}'.format(int(args.target_frame[0]))

print('Volume: %s' % (vvv))
print('Scene: %s' % (vvv))
print('Camera trajectory: %s' % (cam))
print('Source frame: %s' % (s))
print('Target frame: %s' % (t))
print('Occlusion threshold: %.3f' % args.occlusion_threshold)
occlusion_threshold = args.occlusion_threshold
print('Reflectance threshold: %.3f' % args.reflectance_threshold)
reflectance_threshold = args.reflectance_threshold

data_folder = args.data_folder[0] + '/ai_' + vvv + '_' + nnn + '/'

# Images.
data = DataLoader(args.data_folder[0], image_type='jpg', verbose = True)
img_source = data.loadBgr(args.volume_number[0], args.scene_number[0], args.cam_trajectory[0], args.source_frame[0])
img_source_ref = data.loadReflectance(args.volume_number[0], args.scene_number[0], args.cam_trajectory[0], args.source_frame[0])
img_source_position = data.loadPositionMap(args.volume_number[0], args.scene_number[0], args.cam_trajectory[0], args.source_frame[0])

img_target = data.loadBgr(args.volume_number[0], args.scene_number[0], args.cam_trajectory[0], args.target_frame[0])
img_target_ref = data.loadReflectance(args.volume_number[0], args.scene_number[0], args.cam_trajectory[0], args.target_frame[0])
img_target_position = data.loadPositionMap(args.volume_number[0], args.scene_number[0], args.cam_trajectory[0], args.target_frame[0])

# Camera poses.
R_CW, t_CW = data.loadCamPose(args.volume_number[0], args.scene_number[0], args.cam_trajectory[0], args.target_frame[0])

# Construct camera projection matrix
# https://github.com/apple/ml-hypersim/blob/7bc2a8a751c0157c1bd956972acbdd2ddf85186c/code/python/tools/scene_generate_images_bounding_box.py#L129-L149
height_pixels = img_source_position.shape[0]
width_pixels = img_source_position.shape[1]
print('Image height: %d' % (height_pixels))
print('Image width: %d' % (width_pixels))
reprojection = Reprojection()

cv2.imshow('image_source', img_source)
cv2.imshow('image_target', img_target)
cv2.setMouseCallback('image_source', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
