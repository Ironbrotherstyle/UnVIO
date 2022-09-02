# coding=utf-8
import torch.utils.data as data
import sys
sys.path.append("..")
from imageio import imread
from path import Path
import random
import pandas as pd
import time
from utils.pose_transfer import *

train_set = {'2011_10_03_drive_0042_sync_02':'01',
            '2011_10_03_drive_0034_sync_02':'02',
            '2011_10_03_drive_0027_sync_02':'00',
            '2011_09_30_drive_0028_sync_02':'08',
            '2011_09_30_drive_0027_sync_02':'07',
            '2011_09_30_drive_0020_sync_02':'06',
            '2011_09_30_drive_0018_sync_02':'05',
            '2011_09_30_drive_0016_sync_02':'04'
            }
test_set = {'2011_09_30_drive_0033_sync_02':'09',
            '2011_09_30_drive_0034_sync_02':'10'}

def load_as_float(path):
    return imread(path).astype(np.float32)

def pose12to16(mat):
    if mat.ndim == 1:
        mat = mat.reshape(3, -1)
        mat = np.vstack([mat, [0, 0, 0, 1]])
        return mat
    else:
        mat = np.vstack([mat, [0, 0, 0, 1]])
        return mat

def mat_to_6dof(mat):
    if mat.shape[0] == 3:
        mat = pose12to16(mat)
    else:
        translation = list(mat[:3,3])
        rotation = list(euler_from_matrix(mat))
        pose = rotation + translation
    return pose

def absolute2Relative(seqGT):
    sequence_length = len(seqGT)
    seqGT_mat = [pose12to16(item) for item in seqGT]
    seqGT_Rela_mat = []
    seqGT_Rela_mat.append(seqGT_mat[0])
    seqGT_Rela_Eul = []
    seqGT_Rela_Eul.append(mat_to_6dof(seqGT_mat[0]))
    for i in range(1, sequence_length):
        seqGT_Rela_mat.append(np.linalg.inv(seqGT_mat[i-1]) @ seqGT_mat[i])
    seqGT_Rela_mat = np.array(seqGT_Rela_mat)
    return seqGT_Rela_mat

class DataSequence(data.Dataset):
    def __init__(self, root,
                 seed=None,
                 train=True,
                 sequence_length=3,
                 imu_range=[0, 0],
                 transform=None,
                 shuffle=True,
                 scene='default',
                 image_width=640,
                 image_height=480):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.settype = scene
        if self.settype == 'default':
            scene_list = train_set if train == True else test_set
        else:
            scene_list = scene
        self.scenes = [self.root/folder for folder in scene_list]
        self.transform = transform
        self.imu_range = imu_range
        self.shuffle = shuffle
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length+1))

        for scene in self.scenes:
            imgs = sorted((scene).files('*.jpg'))
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))   
            try:
                imus = np.load(scene/'sampled_imu_{}_{}.npy'.format(self.imu_range[0], self.imu_range[1]), allow_pickle=True).astype(np.float32)
            except EOFError as e:
                print("No npy files 'sampled_imu_{}_{}.npy' as commmand specified".format(self.imu_range[0], self.imu_range[1]))
            GT = np.array(pd.read_csv(scene/'poses.txt', sep=' ', header=None))

            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'imgs':[], 'imus':[], 'intrinsics': intrinsics, 'gts': []}
                seq_GT = []
                # put the target in the middle
                for j in shifts:
                    sample['imgs'].append(imgs[i+j])
                    sample['imus'].append(imus[i+j])
                    seq_GT.append(GT[i+j])
                seq_GT = absolute2Relative(seq_GT)
                sample['gts'].append(seq_GT)
                sequence_set.append(sample)

        if self.shuffle:
            random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = [load_as_float(img) for img in sample['imgs']]
        gts = np.squeeze(np.array(sample['gts'])).astype(np.float32)
        if self.transform is not None:
            imgs, imus, intrinsics = self.transform(imgs, np.copy(sample['imus']), np.copy(sample['intrinsics']))
        else:
            intrinsics = np.copy(sample['intrinsics'])
            imus = np.copy(sample['imus'])
        if self.settype == 'default':
            return imgs, imus, intrinsics, np.linalg.inv(intrinsics)
        else:
            return imgs, imus, intrinsics, gts

    def __len__(self):
        return len(self.samples)

if __name__ == "__main__":
    start_time = time.time()
    D = DataSequence(
        sequence_length=5,
        root='/data3/Datasets/weip/KITTI/KITTI_rec_256',
        imu_range=[-10, 0],
        shuffle=False,
        image_width=832,
        image_height=256,
        scene=['2011_09_30_drive_0033_sync_02']
    )
    img, imu, intr, gt = D[0]
    print('time used {}'.format(time.time()-start_time))
