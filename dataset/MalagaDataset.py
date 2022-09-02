# coding=utf-8
import torch.utils.data as data
import sys
sys.path.append("..")
from imageio import imread
from path import Path
import random
import cv2
import time
from utils.pose_transfer import *

train_set = ['01', '02', '04', '05', '06', '08']
test_set = ['03', '07', '09']

def load_as_float(path, width, height):
    return cv2.resize(imread(path), (width, height)).astype(np.float32)

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
        seqGT_Rela_Eul.append(mat_to_6dof(np.linalg.inv(seqGT_mat[i-1]) @ seqGT_mat[i]))
    seqGT_Rela_Eul = np.array(seqGT_Rela_Eul)

    return seqGT_Rela_Eul

class DataSequence(data.Dataset):

    def __init__(self,
                 root,
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
        if scene == 'default':
            scene_list = train_set if train == True else test_set
        else:
            scene_list = scene
        self.scenes = [self.root/folder for folder in scene_list]
        self.transform = transform
        self.imu_range = imu_range
        self.shuffle = shuffle
        self.image_height = image_height
        self.image_width = image_width
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length+1))

        for scene in self.scenes:
            imgs = sorted((scene/'left_256').files('*.jpg'))
            intrinsics = np.genfromtxt(scene/'left_256'/'cam.txt').astype(np.float32).reshape((3, 3)) 
            intrinsics[0, :] *= (self.image_width/832)
            intrinsics[1, :] *= (self.image_height/256)
            try:
                imus = np.load(scene/'imu/sampled_imu_{}_{}.npy'.format(self.imu_range[0],self.imu_range[1])).astype(np.float32)
            except EOFError as e:
                print("No npy files 'sampled_imu_{}_{}.npy' as commmand specified".format(self.imu_range[0],self.imu_range[1]))
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'imgs': [], 'imus': [], 'intrinsics': intrinsics}
                # put the target in the middle
                for j in shifts:
                    sample['imgs'].append(imgs[i+j])
                    sample['imus'].append(imus[i+j])
                sequence_set.append(sample)

        if self.shuffle:
            random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = [load_as_float(img, self.image_width, self.image_height) for img in sample['imgs']]
        if self.transform is not None:
            imgs, imus, intrinsics = self.transform(imgs,  np.copy(sample['imus']),  np.copy(sample['intrinsics']))
        else:
            intrinsics = np.copy(sample['intrinsics'])
            imus = np.copy(sample['imus'])
        return imgs, imus, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)

if __name__ == "__main__":
    start_time = time.time()
    D = DataSequence(
        sequence_length=5,
        root='/data3/Datasets/weip/Malaga/Malaga_Down/',
        imu_range=[-10, 0],
        shuffle=False,
        image_width=832,
        image_height=256,
        train=True,
    )
    img, imu, intr, inr_inv = D[390]
    print('time used {}'.format(time.time()-start_time))