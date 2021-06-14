import numpy as np
import torch

class Reprojection:
    def __init__(self, width=1024, height=768, verbose=False):
        self.verbose = verbose
        self.H = height
        self.W = width
        self.K = self.computeProjectionMatrix()

        if torch.cuda.is_available():
          dev = "cuda:0"
        else:
          dev = "cpu"
        self.device = torch.device(dev)

        if self.verbose:
            print('Device: {:s}'.format(dev))

    def maskFov(self, inliers, px):
        inliers = torch.logical_and(inliers, px[:,:,0].ge(0))
        inliers = torch.logical_and(inliers, px[:,:,1].ge(0))
        inliers = torch.logical_and(inliers, px[:,:,0].lt(self.H))
        inliers = torch.logical_and(inliers, px[:,:,1].lt(self.W))
        return inliers

    # Warp source pixels to target pixels using torch.
    # Input: First pixel entry height, second pixel entry width.
    #        map from pixels to world positions of source frame
    #        rotation of target frame camera
    #        translation of target frame camera
    #        (optional) boolean to mask pixel that are out of field of view of target cam
    #        (optional) map from pixels to world positions of target frame to mask occluded pixel
    #        (optional) occlusion threshold in meters
    #        (optional) map from pixels to reflectance values of source frame to mask reflectant pixel
    #        (optional) reflectance threshold
    # Output: All source pixels, all projected target pixels, list of inliers.
    def warp(self, px_source, source_position_map, R_CW, C_t_CW, mask_fov=False, mask_occlusion=None, occlusion_threshold=0.03, mask_reflectance=None, reflectance_threshold=10, delta_h=0, delta_w=0):
        # Convert input to torch.
        px_source = px_source.to(self.device)
        source_position_map = torch.from_numpy(source_position_map).float().to(self.device)
        K = torch.from_numpy(self.K).float().to(self.device)

        # Construct projection matrix.
        C_T_CW = torch.eye(4).to(self.device)
        C_T_CW[0:3, 0:3] = torch.from_numpy(R_CW).float()
        C_T_CW[0:3, 3] = torch.from_numpy(C_t_CW).float()
        P = torch.matmul(K, C_T_CW).to(self.device)

        # Get homogeneous world position of pixels.
        W_t_WP = source_position_map[px_source[:,:,0], px_source[:,:,1]].to(self.device)
        H, W, C = px_source.shape
        W_t_WP = torch.cat((W_t_WP, torch.ones(H,W,1).to(self.device)), 2).to(self.device)

        # Project pixels into screen coordinates.
        p_screen = torch.matmul(W_t_WP, P.T).to(self.device)

        # Normalize.
        p_screen = torch.div(p_screen, p_screen[:,:,3].view(H,W,1)).to(self.device)

        # Compute pixel coordinates from relative points around camera center.
        px_target = torch.zeros(H,W,C).to(self.device)
        px_target[:,:,1] = 0.5 * (p_screen[:,:,0] + 1) * (self.W - 1)
        px_target[:,:,0] = (1 - 0.5 * (p_screen[:,:,1] + 1)) * (self.H - 1)

        # Round to integer
        px_target = px_target.round().long()
        px_target[:,:,0] += delta_h
        px_target[:,:,1] += delta_w

        # Masking
        # WARNING: The order of the checks matters!
        inlier_mask = torch.ones(H,W, dtype=bool).to(self.device)

        # Mask pixel that are outside of field of view.
        if mask_fov or mask_occlusion is not None:
            inlier_mask = self.maskFov(inlier_mask, px_target)

        # Mask pixel that are occluded.
        if mask_occlusion is not None:
            # Get world position of target pixels that are in FOV.
            target_position_map = torch.from_numpy(mask_occlusion).float().to(self.device)
            W_t_WP_source = source_position_map[px_source[inlier_mask][:,0],px_source[inlier_mask][:,1]].to(self.device)
            W_t_WP_target = target_position_map[px_target[inlier_mask][:,0],px_target[inlier_mask][:,1]].to(self.device)
            inlier_mask[inlier_mask==True] = torch.logical_and(inlier_mask[inlier_mask==True], torch.norm(W_t_WP_target - W_t_WP_source, dim=1).lt(occlusion_threshold))

        # Mask pixel that are reflectant.
        if mask_reflectance is not None:
            source_reflectance_map = torch.from_numpy(mask_reflectance).float().to(self.device)
            reflectance = source_reflectance_map[px_source[inlier_mask][:,0],px_source[inlier_mask][:,1]].to(self.device)
            inlier_mask[inlier_mask==True] = torch.logical_and(inlier_mask[inlier_mask==True], torch.any(reflectance.ge(reflectance_threshold), dim=1))

        return px_source, px_target, inlier_mask

    # Return camera matrix K in 4x4 to project 3D coordinates in camera frame to image plane.
    # https://github.com/apple/ml-hypersim/blob/7bc2a8a751c0157c1bd956972acbdd2ddf85186c/code/python/tools/scene_generate_images_bounding_box.py#L129-L149
    def computeProjectionMatrix(self):
        fov_x = np.pi / 3.0
        fov_y = 2.0 * np.arctan(self.H * np.tan(fov_x / 2.0) / self.W)
        near = 1.0
        far = 1000.0

        f_h = np.tan(fov_y / 2.0) * near
        f_w = f_h * self.W / self.H
        left = -f_w
        right = f_w
        bottom = -f_h
        top = f_h

        K = np.matrix(np.zeros((4, 4)))
        K[0, 0] = (2.0 * near) / (right - left)
        K[1, 1] = (2.0 * near) / (top - bottom)
        K[0, 2] = (right + left) / (right - left)
        K[1, 2] = (top + bottom) / (top - bottom)
        K[2, 2] = -(far + near) / (far - near)
        K[3, 2] = -1.0
        K[2, 3] = -(2.0 * far * near) / (far - near)
        if self.verbose:
            print('Camera projection matrix:')
            print(K)
        return K
