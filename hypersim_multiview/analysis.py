import torch
from matplotlib import pyplot as plt
import cv2
import os
import csv

class Analysis:
    def __init__(self, px_source, px_target, inliers, reprojection, \
                    warp_time, result_path, vol, scene, cam, \
                    source_frame, target_frame, verbose=False):
        self.verbose = verbose
        self.px_source = px_source
        self.px_target = px_target
        self.inliers = inliers
        self.reprojection = reprojection
        self.warp_time = warp_time
        self.result_path = result_path
        self.vol = vol
        self.scene = scene
        self.cam = cam
        self.source_frame = source_frame
        self.target_frame = target_frame

        # Cache analysis results.
        self.dif = None
        self.delta_h = None
        self.delta_w = None
        self.error = None
        self.min_error = None
        self.max_error = None
        self.median_error = None
        self.mean_error = None
        self.std_error = None

    def displayFrames(self, img_source, img_target):
        cv2.imshow('Source Image', img_source)
        cv2.imshow('Target Image', img_target)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def maskImage(self, img, px):
        blue = torch.tensor([255, 0, 0], dtype=torch.uint8)
        red = torch.tensor([0, 0, 255], dtype=torch.uint8)

        # Slicing to make underlying picture visible.
        print
        H, W = self.inliers.shape
        inliers = self.inliers[torch.arange(0, H, 2),:]
        inliers = inliers[:,torch.arange(0, W, 2)]
        px = px[torch.arange(0, H, 2),:,:]
        px = px[:,torch.arange(0, W, 2),:]

        # Double check if in FOV.
        outliers = ~inliers

        inliers = self.reprojection.maskFov(inliers, px)
        outliers = self.reprojection.maskFov(outliers, px)

        px_inlier = px[inliers][:,0].cpu(),px[inliers][:,1].cpu()
        px_outlier = px[outliers][:,0].cpu(),px[outliers][:,1].cpu()
        # Print all inliers.
        masked_img = img.copy()
        masked_img[px_inlier]=0
        masked_img[px_outlier]=0
        masked_img[px_inlier]+=blue.numpy()
        masked_img[px_outlier]+=red.numpy()

        return masked_img

    def displayMaskedFrames(self, img_source, img_target):
        img_source = self.maskImage(img_source, self.px_source)
        img_target = self.maskImage(img_target, self.px_target)

        cv2.imshow('Source Image', img_source)
        cv2.imshow('Target Image', img_target)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def saveMaskedFrames(self, img_source, img_target):
        img_source = self.maskImage(img_source, self.px_source)
        img_target = self.maskImage(img_target, self.px_target)
        folder = os.path.join(self.result_path, 'ai_{:03d}_{:03d}/images/scene_cam_{:02d}_reprojection_analysis'.format(self.vol, self.scene, self.cam))
        if not os.path.exists(folder):
            os.makedirs(folder)
        source_name = 'frame_{:04d}.source.reprojection.from_frame_{:04d}_to_frame_{:04d}.jpg'.format(self.source_frame, self.source_frame, self.target_frame)
        target_name = 'frame_{:04d}.target.reprojection.from_frame_{:04d}_to_frame_{:04d}.jpg'.format(self.target_frame, self.source_frame, self.target_frame)
        source_file = os.path.join(folder, source_name)
        target_file = os.path.join(folder, target_name)
        if self.verbose:
            print('Saving masked source frame image to {:s}'.format(source_file, target_file))
            print('Saving masked target frame image to {:s}'.format(target_file))
        cv2.imwrite(source_file, img_source)
        cv2.imwrite(target_file, img_target)

    def computeStatistics(self):
        if self.source_frame != self.target_frame:
            print("WARNING: Statistics only make sense for identical source and target frame!")

        refresh = False
        if self.dif is None:
            refresh = True
            self.dif = self.px_target - self.px_source
        if self.delta_h is None:
            self.delta_h = -torch.median(self.dif[:,:,0])
        if self.delta_w is None:
            self.delta_w = -torch.median(self.dif[:,:,1])
        if self.error is None:
            self.error = torch.linalg.norm(self.dif.float(), dim=2)
        if self.min_error is None:
            self.min_error = torch.min(self.error)
        if self.max_error is None:
            self.max_error = torch.max(self.error)
        if self.median_error is None:
            self.median_error = torch.median(self.error)
        if self.mean_error is None and self.std_error is None:
            self.std_error, self.mean_error = torch.std_mean(self.error)


        if self.verbose and refresh:
            print('Delta H: {:d}'.format(self.delta_h))
            print('Delta W: {:d}'.format(self.delta_w))
            print('Min pixel error: {:.1f}'.format(self.min_error))
            print('Max pixel error: {:.1f}'.format(self.max_error))
            print('Median pixel error: {:.1f}'.format(self.median_error))
            print('Mean pixel error: {:.1f}'.format(self.mean_error))
            print('Std pixel error: {:.1f}'.format(self.std_error))

    def plotErrorHistogram(self, cutoff=100):
        self.computeStatistics()

        H, W, _ = self.px_source.shape
        plt.figure()
        error = self.error.view(H*W)
        error[error > cutoff] = cutoff
        plt.hist(self.error.view(H*W).cpu().numpy().flatten())
        plt.title('Volume {:d}, Scene {:d}, Camera {:d}, Frame {:d}\nMin. {:.1f} px, Max. {:.1f} px, Median {:.1f} px, Mean {:.1f} px, Std. Dev. {:.1f} px'.format(\
                    self.vol, self.scene, self.cam, self.source_frame, \
                    self.min_error, self.max_error, self.median_error, self.mean_error, self.std_error))
        plt.xlabel('Reprojection Error [px]')
        plt.ylabel('Number of Pixels')
        plt.grid()
        folder = os.path.join(self.result_path, 'ai_{:03d}_{:03d}/images/scene_cam_{:02d}_reprojection_analysis'.format(self.vol, self.scene, self.cam))
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = os.path.join(folder, 'frame_{:04d}.reprojection_error.pdf'.format(self.source_frame))
        if self.verbose:
            print('Saving error histogram to {:s}'.format(file))
        plt.savefig(os.path.abspath(file))
        plt.close()

    def saveStatistics(self):
        self.computeStatistics()

        file = os.path.join(self.result_path, 'error_statistics.csv')

        if self.verbose:
            print('Saving statistcs in {:s}'.format(file))

        fieldnames = ['scene_name', 'camera_name', 'frame_id', 'min_error', 'max_error', 'median_error', 'mean_error', 'std_error', 'delta_h', 'delta_w', 'warp_time']
        row = {'scene_name': 'ai_{:03d}_{:03d}'.format(self.vol, self.scene), \
               'camera_name': 'cam_{:02d}'.format(self.cam), \
               'frame_id': self.source_frame, \
               'min_error': self.min_error.cpu().numpy(), \
               'max_error': self.max_error.cpu().numpy(), \
               'median_error': self.median_error.cpu().numpy(), \
               'mean_error': self.mean_error.cpu().numpy(), \
               'std_error': self.std_error.cpu().numpy(), \
               'delta_h': self.delta_h.cpu().numpy(), \
               'delta_w': self.delta_w.cpu().numpy(), \
               'warp_time': self.warp_time}

        if not os.path.isfile(file):
            with open(file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
        else:
            with open(file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(row)
