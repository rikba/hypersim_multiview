import cv2
import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt

from vision_utils.data_loader import DataLoader
from kp2d.networks.keypoint_net import KeypointNet
from kp2d.networks.keypoint_resnet import KeypointResnet
from kp2d.utils.keypoints import draw_keypoints

def main():
    parser = argparse.ArgumentParser(
        description='Apply network to an image.')
    parser.add_argument('model', nargs=1,
                        help='The pretrained image network.')
    parser.add_argument('data_folder', nargs=1,
                        help='The folder containing the hypersim scenes.')
    parser.add_argument('volume_number', nargs=1, help='The volume number.')
    parser.add_argument('scene_number', nargs=1, help='The scene number.')
    parser.add_argument('cam_trajectory', nargs=1,
                        help='The camera trajectory number.')
    parser.add_argument('frame', nargs=1, help='The image frame.')
    args = parser.parse_args()

    checkpoint = torch.load(args.model[0])
    model_args = checkpoint['config']['model']['params']

    # Check model type
    if 'keypoint_net_type' in checkpoint['config']['model']['params']:
       net_type = checkpoint['config']['model']['params']
    else:
       net_type = KeypointNet # default when no type is specified

    # Create and load keypoint net
    if net_type is KeypointNet:
       keypoint_net = KeypointNet(use_color=model_args['use_color'],
                               do_upsample=model_args['do_upsample'],
                               do_cross=model_args['do_cross'])
    else:
       keypoint_net = KeypointResnet()

    keypoint_net.load_state_dict(checkpoint['state_dict'])
    keypoint_net = keypoint_net.cuda()
    keypoint_net.eval()
    print('Loaded KeypointNet from {}'.format(args.model[0]))
    print('KeypointNet params {}'.format(model_args))

    data = DataLoader(args.data_folder[0], verbose = True)
    img_orig = data.loadRgb(args.volume_number[0], args.scene_number[0], args.cam_trajectory[0], args.frame[0])
    img2_orig = data.loadRgb(args.volume_number[0], args.scene_number[0], args.cam_trajectory[0], int(args.frame[0])+1)

    test = np.copy(img_orig)
    test[test<=1] = 0
    test[test>1] = 1
    plt.imshow(np.float32(test))
    plt.show()

    img_clipped = np.copy(img_orig)
    img_clipped[img_clipped>1] = 1

    img2_clipped = np.copy(img2_orig)
    img2_clipped[img2_clipped>1] = 1

    for img in [img_orig, img_clipped, img2_clipped]:
        transform = transforms.ToTensor()
        tensor = transform(img).type('torch.FloatTensor').cuda()
        score, uv_pred, feat = keypoint_net.forward(tensor.unsqueeze(0))

        print(tensor.shape)
        print(score.shape)
        print(uv_pred.shape)
        print(feat.shape)

        top_k2 = 100
        _, top_k = score.view(1,-1).topk(top_k2, dim=1)
        keypoints = uv_pred.view(1,2,-1)[:,:,top_k[0].squeeze()]
        keypoints = keypoints.permute(0, 2, 1)[0].detach().cpu().clone().numpy()
        #print(keypoints)



        #print(top_k)

        #vis_xyd = uv_pred.permute(0, 2, 1)[idx].detach().cpu().clone().numpy()

        #plt.imshow(  score.squeeze().permute(1, 2, 0)  )
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(  np.float32(img)  )
        ax2.imshow(  score.cpu().detach().numpy().squeeze()  )
        for keypoint in keypoints:
            circle = plt.Circle((keypoint[0], keypoint[1]), 5, color='r')
            ax1.add_artist(circle)
        plt.show()

if __name__ == '__main__':
    main()
