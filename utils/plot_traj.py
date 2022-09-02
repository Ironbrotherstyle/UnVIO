import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

def pose12to16(mat):
    if mat.ndim == 1:
        mat = mat.reshape(3, -1)
        mat = np.vstack([mat, [0, 0, 0, 1]])
        return mat
    else:
        mat = np.vstack([mat, [0, 0, 0, 1]])
        return mat
def pose16to12(mat):
    return (mat[:3][:3])

def plot_route3d(gt_coords, pred_coords, c_gt='r', c_out='b'):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title("{}".format(1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if gt_coords:
        x0, y0, z0 = gt_coords
        ax.plot(x0, y0, z0, color=c_gt, label='Ground Truth')
    if pred_coords:
        x1, y1, z1 = pred_coords
        ax.plot(x1, y1, z1, color=c_out, label='UnVIO')
    plt.show()

def plot_route2d(gt_coords, pred_coords, c_gt='r', c_out='b'):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title("{}".format(1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if gt_coords:
        x0, y0, z0 = gt_coords
        ax.plot(x0, z0, color=c_gt, label='Ground Truth')
    if pred_coords:
        x1, y1, z1 = pred_coords
        ax.plot(x1, z1, color=c_out, label='UnVIO')
    plt.show()

def get_prediction_traj(predictions):
    X = predictions[:, 0, 3]
    Y = predictions[:, 1, 3]
    Z = predictions[:, 2, 3]
    return X, Y, Z

def get_gt_traj(groundtruth):
    X = groundtruth[:,3]
    Y = groundtruth[:,7]
    Z = groundtruth[:,11]
    return X, Y, Z

if __name__ == '__main__':
    scale = 1
    groundtruth = np.loadtxt('sampled_12Dpose.csv',delimiter=',',dtype=np.float32)
    predictions = np.load('predpose_09.npy')
    gt_coords = get_gt_traj(groundtruth)
    pred_coords = get_prediction_traj(predictions*scale)
    plot_route3d(None, pred_coords)
    plt.show()

