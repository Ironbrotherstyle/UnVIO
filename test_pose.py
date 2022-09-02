import torch
import models
import os
from utils import custom_transform
from utils.pose_transfer import *
from utils.inverse_warp import *
import argparse
from utils.plot_traj import get_prediction_traj, plot_route3d, plot_route2d
import pandas as pd

parser = argparse.ArgumentParser(description='UnVIO test file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset_root', type=str, help='dataset root path')
parser.add_argument('--dataset', type=str, choices=['KITTI', 'Malaga'], default='KITTI', help='which dataset is used to test')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for testing', default=5)
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler', help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser.add_argument('--pretrained-visualnet', dest='pretrained_visualnet', default=None, metavar='PATH', help='path to pre-trained visual net model')
parser.add_argument('--pretrained-imunet', dest='pretrained_imunet', default=None, metavar='PATH', help='path to pre-trained imu net model')
parser.add_argument('--pretrained-posenet', dest='pretrained_posenet', default=None, metavar='PATH', help='path to pre-trained pose net model')
parser.add_argument('--testscene', type=str, help='test_scene of dataset')
parser.add_argument("--show-traj", action='store_true', default=False, help="show trajectory")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

args = parser.parse_args()

@torch.no_grad()
def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    data_root = args.dataset_root
    args.imu_range = [-10, 0]
    # load data
    if args.dataset == 'KITTI':
        from dataset.KITTIDataset import DataSequence as dataset
        data_path = '{}/KITTI_rec_256'.format(data_root)
        args.img_width = 832
        args.img_height = 256
    elif args.dataset == 'Malaga':
        from dataset.MalagaDataset import DataSequence as dataset
        data_path = '{}/Malaga_Down/'.format(data_root)
        args.img_width = 832
        args.img_height = 256
    else:
        raise ValueError('Unsupported dataset')

    # init visual net, imunet, posenet
    visual_net = models.models.VisualNet().to(device)
    imu_net = models.models.ImuNet().to(device)
    pose_net = models.models.PoseNet(input_size=1024).to(device)

    if args.pretrained_posenet:
        print("=> using pre-trained weights for pose_net")
        weights = torch.load(args.pretrained_posenet)
        pose_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        pose_net.init_weights()

    if args.pretrained_visualnet:
        print("=> using pre-trained weights for visual_net")
        weights = torch.load(args.pretrained_visualnet)
        visual_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        visual_net.init_weights()

    if args.pretrained_imunet:
        print("=> using pre-trained weights for imu_net")
        weights = torch.load(args.pretrained_imunet)
        imu_net.load_state_dict(weights['state_dict'], strict=True)
    else:
        imu_net.init_weights()

    test_scene = [args.testscene]
    print('test_scene: ', test_scene)

    save_path = os.path.split(args.pretrained_posenet)[0]
    print('=> Save results at {}'.format(save_path))

    test_transform = custom_transform.Compose([
        custom_transform.ToTensor(),
        custom_transform.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_set = dataset(root=data_path,
                       seed=0,
                       sequence_length=args.sequence_length,
                       imu_range=args.imu_range,
                       transform=test_transform,
                       shuffle=False,
                       scene=test_scene,
                       image_width=args.img_width,
                       image_height=args.img_height)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True)

    print('sequence length: {}'.format(len(test_loader)))
    predictions_array = []
    visual_net.eval()
    imu_net.eval()
    pose_net.eval()
    GTS = []

    for i, (imgs, imus, intr, gt) in enumerate(test_loader):
        if args.dataset == 'KITTI':
            if i == 0:
                GTS.append(gt[0, 0])
                for j in range(1, args.sequence_length):
                    GTS.append(GTS[-1] @ gt[0, j])

            else:
                GTS.append(GTS[-1] @ gt[0, -1])

        print('{}th batch of total {}'.format(i + 1, len(test_loader)), end="")
        print('\r', end="", flush=True)

        imgs = [img.to(device) for img in imgs]
        imus = imus.to(device)
        visual_feature = visual_net(imgs)  # B T 1024
        imu_feature = imu_net(imus[:, 1:])
        out = pose_net(visual_feature,
                           imu_feature).data.cpu().numpy()  # B T-1 6

        if i == 0:
            predictions_array.append(np.zeros([1, 6]))
            for j in range(args.sequence_length - 1):
                predictions_array.append(out[:, j, :])

        else:
            predictions_array.append(out[:, -1, :])

    print('\nlength of predictions_array: ', len(predictions_array))
    absolute_pose = np.array(relative2absolute(predictions_array))
    tmppose = absolute_pose[:, :3].reshape(absolute_pose.shape[0], -1)
    save_csv = pd.DataFrame(tmppose)
    save_csv.to_csv(save_path +
                    '/predpose_{}.csv'.format(''.join(test_scene)),
                    header=None,
                    index=None)

    if args.show_traj:
        if args.dataset == 'KITTI':
            GTS = np.stack(GTS, 0)
            scale = GTS[:, :3, 3].mean() / absolute_pose[:, :3, 3].mean()
            plot_route2d(get_prediction_traj(GTS),
                         get_prediction_traj(absolute_pose * scale))
        else:
            plot_route3d(None, get_prediction_traj(absolute_pose))

def out2pose(out):
    pose = [pose_vec2mat4(out[:, i]) for i in range(4)]
    pose = [
        pose[0] @ pose[1], pose[1],
        b_inv(pose[2]),
        b_inv(pose[2] @ pose[3])
    ]
    pose = torch.stack(pose, dim=1)
    return pose[:, :, :3, :]

def relative2absolute(pose):
    abs_pose_mat = []
    for i in range(len(pose)):
        temp_mat = _6Dofto16mat(pose[i])
        if i == 0:
            abs_pose_mat.append(temp_mat)
        else:
            abs_pose_mat.append(abs_pose_mat[i - 1] @ temp_mat)
    return abs_pose_mat

def _6Dofto16mat(pose):
    translation = pose[0][3:]
    rotation = pose[0][:3]
    R = euler_matrix(rotation[0], rotation[1], rotation[2])
    T = np.vstack([np.hstack([R, translation.reshape(-1, 1)]), [0, 0, 0, 1]])
    return T

def _16mat26Dof(mat):
    translation = [str(item) for item in list(mat[:3, 3])]
    roation = [str(item) for item in list(euler_from_matrix(mat))]
    return roation + translation

if __name__ == "__main__":
    main()
