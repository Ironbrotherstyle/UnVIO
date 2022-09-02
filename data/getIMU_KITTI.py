# coding=utf-8
'''
generate IMU data corresponding to image timestamp, which will be saved as .npy file
'''
import pandas as pd
import numpy as np

def time2timestamp(time_):
    time1 = time_.split('.')[0]
    time2 = time_.split('.')[1]
    h, m, s = time1.split(':')

    integer = str(int(h)*3600 + int(m)*60 + int(s))
    decimal = time2
    
    time_stamp = int(integer+decimal)
    return time_stamp

def getClosestIndex(file_time, original_imu):
    foundIdx = 0
    for i in range(0, len(original_imu)):
        if time2timestamp(original_imu[i][0]) >= file_time:
            foundIdx = i
            break
    return foundIdx

def get_IMUS(dataset_dir, before, after):
    for scene in dataset:
        scene_dir = dataset_dir + '/' + scene
        filecsv = np.array(pd.read_csv(scene_dir + '/oxts.csv'))

        # read original IMU data
        original_imu = np.array(pd.read_csv(scene_dir + '/oxts_ori.csv'))

        sampled_imu = []
        for i in range(len(filecsv)):
            index_imu = []
            file_time = time2timestamp(filecsv[i][0])
            foundIdx = getClosestIndex(file_time, original_imu)
            for j in range(before, after+1):
                if foundIdx+j <= len(original_imu)-1:
                    temp_imu = original_imu[foundIdx+j][1:]
                else:
                    temp_imu = original_imu[len(original_imu)-1][1:]
                index_imu.append(temp_imu)
            sampled_imu.append(index_imu)
        
        sampled_imu = np.array(sampled_imu)

        np.save(scene_dir + '/sampled_imu_{}_{}.npy'.format(before, after), sampled_imu)

    return 

if __name__ == "__main__":
    dataset = {'2011_10_03_drive_0042_sync_02':'01',
                '2011_10_03_drive_0034_sync_02':'02',
                '2011_10_03_drive_0027_sync_02':'00',
                '2011_09_30_drive_0033_sync_02':'09',
                '2011_09_30_drive_0028_sync_02':'08',
                '2011_09_30_drive_0027_sync_02':'07',
                '2011_09_30_drive_0020_sync_02':'06',
                '2011_09_30_drive_0018_sync_02':'05',
                '2011_09_30_drive_0016_sync_02':'04',
                '2011_09_30_drive_0034_sync_02':'10'}

    dataset_dir = '/data3/Datasets/weip/KITTI/KITTI_rec_256'
    get_IMUS(dataset_dir, -10, 0)
