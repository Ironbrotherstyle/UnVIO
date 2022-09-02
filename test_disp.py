import argparse
import os
import cv2
import numpy as np
import torch
from path import Path
from scipy.misc import imread, imresize
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import matplotlib.pyplot as plt

import models
cmap = plt.cm.magma

parser = argparse.ArgumentParser(description='Script for DispNet testing with corresponding groundTruth', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-dispnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--save-depth", action='store_true', help="if save depth map")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--gt-type", default='KITTI', type=str, help="GroundTruth data type", choices=['npy', 'png', 'KITTI', 'stillbox'])
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_tensor_image(filename, args):
    img = imread(filename).astype(np.float32)
    h, w, _ = img.shape
    if (not args.no_resize) and (h != args.img_height or w != args.img_width):
        img = imresize(img,
                       (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0) / 255 - 0.5) /
                  0.5).to(device)
    return tensor_img

def colored_depth(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = ((depth-d_min)/(d_max-d_min))
    return 255 * cmap(depth_relative)[:,:,:3]

@torch.no_grad()
def main():
    args = parser.parse_args()
    disp_net = models.DepthDecoder(alpha=10, beta=0.01).to(device)
    weights = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    if args.dataset_list:
        dataset_dir = Path(args.dataset_dir)
        with open(args.dataset_list, 'r') as f:
            test_files = list(f.read().splitlines())
    else:
        print('No dataset_list, please specify the dataset list for depth estimation')
        exit()

    print('{} files to test'.format(len(test_files)))

    if not args.output_dir:
        args.output_dir = args.dataset_dir
    output_dir = Path(args.output_dir)
    print('writing npy file in: ', output_dir)
    output_dir.makedirs_p()

    if args.save_depth:
        output_depth_dir = output_dir/'disp'
        print('saving depth maps in: ', output_depth_dir)
        output_depth_dir.makedirs_p()

    for j in tqdm(range(len(test_files))):
        tgt_img = load_tensor_image(dataset_dir + test_files[j], args)
        pred_disp = disp_net(tgt_img).cpu().numpy()[0, 0]

        if j == 0:
            predictions = np.zeros((len(test_files), *pred_disp.shape))
        predictions[j] = 1 / pred_disp

        if args.save_depth:
            depth_colored = colored_depth(pred_disp)[:, :, ::-1]
            depth_colored = np.array(depth_colored, dtype=np.uint8)
            cv2.imwrite(output_depth_dir/'{}_disp.png'.format(str(j).zfill(6)), depth_colored)

    np.save(output_dir / 'predictions.npy', predictions)

if __name__ == '__main__':
    main()
