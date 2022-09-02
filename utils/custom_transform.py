from __future__ import division
import torch
import random
import numpy as np
from scipy.misc import imresize

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, imus, intrinsics):
        for t in self.transforms:
            images, imus, intrinsics = t(images, imus, intrinsics)
        return images, imus, intrinsics

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, imus, intrinsics):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, imus, intrinsics

class ToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images, imus, intrinsics):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float()/255)
        return tensors, imus, intrinsics

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, imus, intrinsics):
        assert intrinsics is not None
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(im)) for im in images]
            w = output_images[0].shape[1]
            output_intrinsics[0,2] = w - output_intrinsics[0,2]
        else:
            output_images = images
            output_intrinsics = intrinsics
        return output_images, imus, output_intrinsics


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, imus, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1,1.15,2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling
        scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]

        output_intrinsics[0,2] -= offset_x
        output_intrinsics[1,2] -= offset_y

        return cropped_images, imus, output_intrinsics


class AugmentImagePair(object):
    def __init__(self, augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2]):
        self.gamma_low = augment_parameters[0]  # 0.8
        self.gamma_high = augment_parameters[1]  # 1.2
        self.brightness_low = augment_parameters[2]  # 0.5
        self.brightness_high = augment_parameters[3]  # 2.0
        self.color_low = augment_parameters[4]  # 0.8
        self.color_high = augment_parameters[5]  # 1.2

    def __call__(self, images, imus, intrinsics):
        p = np.random.uniform(0, 1, 1)
        sample = []
        if p > 0.5:
            random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
            random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
            random_colors = np.random.uniform(self.color_low, self.color_high, 3)
            for img in images:
                # randomly shift gamma
                img_aug = img ** random_gamma

                # randomly shift brightness
                img_aug = img_aug * random_brightness

                # randomly shift color
                for i in range(3):
                    img_aug[i, :, :] *= random_colors[i]

                # saturate
                img_aug = torch.clamp(img_aug, 0, 1)
                sample.append(img_aug)
        else:
            sample = images
        return sample, imus, intrinsics