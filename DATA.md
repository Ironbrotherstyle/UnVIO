Note that we resize the image both in
KITTI dataset and Malaga dataset from original resolution into $832 \times 256$. 

## KITTI dataset

We list the sequence index of KITTI Odometry split and corresponding folder name
in KITTI raw split (03 is not included in training set as the IMU data is not available):

| Seq index | name                  |
|-----------|-----------------------|
| 00        | 2011_10_03_drive_0027 |
| 01        | 2011_10_03_drive_0042 |
| 02        | 2011_10_03_drive_0034 |
| 03        | 2011_09_26_drive_0067 |
| 04        | 2011_09_30_drive_0016 |
| 05        | 2011_09_30_drive_0018 |
| 06        | 2011_09_30_drive_0020 |
| 07        | 2011_09_30_drive_0027 |
| 08        | 2011_09_30_drive_0028 |
| 09        | 2011_09_30_drive_0033 |
| 10        | 2011_09_30_drive_0034 |

the processed KITTI data for training and validation will be placed in folder `KITTI_rec_256` as
follows

``` 
├── 2011_09_26_drive_0067_sync_02
├── 2011_09_30_drive_0016_sync_02
├── 2011_09_30_drive_0018_sync_02
├── 2011_09_30_drive_0020_sync_02
├── 2011_09_30_drive_0027_sync_02
├── 2011_09_30_drive_0028_sync_02
├── 2011_09_30_drive_0033_sync_02
├── 2011_09_30_drive_0034_sync_02
├── 2011_10_03_drive_0027_sync_02
├── 2011_10_03_drive_0034_sync_02
├── 2011_10_03_drive_0042_sync_02
├── train.txt
└── val.txt
```

There are images, camera intrinsic parameters, sparse imu
values, dense imu values, ground truth poses, sampled imu data
in each sub folder. The structure are listed as follows: 

``` 
├── 000000xxxx.jpg
├── ......
├── cam.txt
├── oxts.csv
├── oxts_ori.csv
├── poses.csv
├── sampled_imu_index1_index2.npy


```

## Malaga dataset

We list the folder structure of Malaga dataset as follows,

``` 
├── 01
├── 02
├── ...
```

in each folder, there are imu data, images, timestamp arranged 
as follows,

``` 
├── imu
├── left_256
├──── xxxxxxxxxx.xxxxxx.jpg
├──── ......
├──── cam.txt
├── data.csv
```