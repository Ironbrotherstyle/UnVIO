# Unsupervised Network for Visual Inertial Odometry
IJCAI2020 paper: Unsupervised Network for Visual Inertial Odometry. 

| KITTI 09  | KITTI 10 |
| ------------- | ------------- |
| ![aa](./imgs/kitti_09_final.gif) | ![bb](./imgs/kitti_10_final.gif) |


## Introduction
This repository is the official [Pytorch](https://pytorch.org/) implementation of IJCAI2020 paper [Unsupervised Network for Visual Inertial Odometry](https://robotics.pkusz.edu.cn/static/papers/IJCAI-weipeng.pdf). 


## Installation

UnVIO has been tested on Ubuntu with Pytorch 1.4 and Python 3.7.10. For installation, it is recommended to use conda environment.


```shell
conda create -n unvio_env python=3.7.10
conda activate unvio_env
pip install -r requirements.txt
```

Other applications should be installed also,
```shell
sudo apt install gnuplot
```

## Data Preparing

The datasets used in this paper are [KITTI raw](http://www.cvlibs.net/datasets/kitti/raw_data.php) ataset 
and [Malaga](https://www.mrpt.org/MalagaUrbanDataset) dataset. Please refer to [Data preparing](DATA.md) for detailed 
instruction.

## Validation

Validation can be implemented on Depth estimation and Odometry estimation.
First specify the model path and dataset path:

```shell
ROOT='MODEL_ROOT_HERE'
DATA_ROOT='DATA_ROOT_HERE'
```

### Depth Estimation

For Depth estimation on KITTI 09 (if you want to test on KITTI 10, change the 
`--dataset-list` to `.eval/kitti_10.txt`), run the following command:

```shell
python test_disp.py \
   --pretrained-dispnet $ROOT/dispnet_checkpoint.pth.tar \
   --dataset-dir $DATA_ROOT \
   --dataset-list .eval/kitti_09.txt \
   --output-dir $ROOT/results_disp \
   --save-depth
```

The `predictions.npy` that stores the all the depth values will be saved in `$ROOT/results_disp`, if `--save-depth` is added, the colored depths will be saved simultaneously is `$ROOT/results_disp/disp`

### Visual Odometry

For Odometry estimation KITTI 09 (if you want to test on KITTI 10, change the `testscene` to `2011_09_30_drive_0034_sync_02`), run the following command: 

```shell
python test_pose.py \
 --pretrained-posefea $ROOT/PoseNet_fea_model_best.pth.tar \
 --pretrained-imu $ROOT/DeepVIO_IMU_model_best.pth.tar\
 --pretrained-pose $ROOT/DeepVIO_POSE_model_best.pth.tar\
 --testscene 2011_09_30_drive_0033_sync_02 \
 --show-traj
```

This will create a `.csv` file represneting $T_{wc} \in \mathbb{R}^{3 \times 4}$ in `$ROOT` directory. If the `--show-traj` is added, a scaled trajectory comparing with the ground truth will be ploted. 

## Train

Run the following command to train the UnVIO from scratch:

```shell
python train.py 
```

specify `--dataset` as you need.

## Citation

```
@inproceedings{2020Unsupervised,
  title={Unsupervised Monocular Visual-inertial Odometry Network},
  author={ Wei, P.  and  Hua, G.  and  Huang, W.  and  Meng, F.  and  Liu, H. },
  booktitle={Twenty-Ninth International Joint Conference on Artificial Intelligence and Seventeenth Pacific Rim International Conference on Artificial Intelligence {IJCAI-PRICAI-20}},
  year={2020},
}
```

## License

This project is licensed under the terms of the MIT license.

## References

The repository borrowed some code from [SC](https://github.com/JiawangBian/SC-SfMLearner-Release), [Monodepth2](https://github.com/nianticlabs/monodepth2.git) and [SfMLearner](https://github.com/ClementPinard/SfmLearner-Pytorch), thanks for their great work.


