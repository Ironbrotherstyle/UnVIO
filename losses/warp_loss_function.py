from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from utils.inverse_warp import inverse_warp_3d
from .InverseDepthSmoothnessLoss import inverse_depth_smoothness_loss

def photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics,
                                    depth, pose, 
                                    rotation_mode='euler', padding_mode='zeros', explainability_mask=None, ref_depth=None):
    def one_scale(depth, explainability_mask = None, ref_depth=None):
        assert(pose.size(1) == len(ref_imgs))

        reconstruction_loss, loss_3d = 0, 0
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h
        tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
        ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]

        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]
            if (ref_depth is not None) and (ref_depth[i] is not None):
                ref_img_warped, coords_3d = inverse_warp_3d(ref_img, depth[:,0], current_pose,
                                          intrinsics_scaled, rotation_mode, padding_mode, ref_depth=ref_depth[i][:, 0])
            else:
                ref_img_warped, _ = inverse_warp_3d(ref_img, depth[:,0], current_pose,
                                          intrinsics_scaled, rotation_mode, padding_mode, ref_depth=None)
            out_of_bound = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)   # B 1 H W
            diff = appearence_loss(tgt_img_scaled, ref_img_warped, out_of_bound)

            if (ref_depth is not None) and (ref_depth[i] is not None):
                loss_3d += (out_of_bound * (coords_3d[0][:,2:] - coords_3d[1][:,2:]).abs() / (coords_3d[0][:,2:] + coords_3d[1][:,2:])).mean()

            if explainability_mask is not None:
                diff = diff * explainability_mask[:,i:i+1].expand_as(diff)

            reconstruction_loss += diff.mean()
            assert((reconstruction_loss == reconstruction_loss).item() == 1)

        return reconstruction_loss/len(ref_imgs), loss_3d/len(ref_imgs)

    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    loss_photo, loss_3d = 0, 0

    for i, d in enumerate(depth):
        r_depth = ref_depth[i]
        tmp1, tmp2 = one_scale(d, None, ref_depth=r_depth)
        loss_photo += tmp1
        loss_3d += tmp2
    return loss_photo / len(depth), loss_3d/len(depth)

def appearence_loss(img1, img2, valid_pixel, ternary=False, weights=[0.15, 0.85, 0.08]):

    diff = img1 - img2
    diff = charbonnier_loss(diff, valid_pixel, alpha=0.5, size_average=False)
    ssim_loss = s_SSIM(img1*valid_pixel, img2*valid_pixel, window_size=3).mean(1, keepdim=True)
    if ternary:
        ternaryloss = ternary_loss(img1, img2, valid_pixel)
    else:
        ternaryloss = 0
    return weights[0] * diff + weights[1] * ssim_loss + weights[2] * ternaryloss

def s_SSIM(x, y, window_size):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    avg = nn.AvgPool2d(kernel_size=window_size, padding=window_size//2, stride=1)
    mu_x = avg(x)
    mu_y = avg(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = avg(x * x) - mu_x_sq
    sigma_y = avg(y * y) - mu_y_sq
    sigma_xy = avg(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    return torch.clamp((1 - SSIM) / 2, 0, 1)

def charbonnier_loss(x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001, size_average=False):
    """Compute the generalized charbonnier loss of the difference tensor x.
    All positions where mask == 0 are not taken into account.
    Args:
        x: a tensor of shape [num_batch, channels, height, width].
        mask: a mask of shape [num_batch, mask_channels, height, width],
            where mask channels must be either 1 or the same number as
            the number of channels of x. Entries should be 0 or 1.
    Returns:
        loss as floatTensor
    """
    if x.size(1) == 1 and mask.size(1) == 3:
        mask = mask.mean(1,keepdim=True)
    assert(x.size(1) == mask.size(1) or mask.size(1) == 1)
    b, c, h, w = x.size()

    error = ((x*beta)**2 + epsilon**2).pow(alpha)

    if mask is not None:
        error = error * mask

    if truncate is not None:
        error = torch.min(error, truncate)
    if size_average:
        return error.mean()
    else:
        return error.mean(1, keepdim=True) 

def explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = torch.ones_like(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss

def smooth_loss(pred_map):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # don't ask me why it works better
    return loss

def disp_smooth_loss(disparities, img):
    loss = 0
    weight = 1.
    for disp in disparities:
        b, _, h, w = disp.size()
        img_scaled = F.interpolate(img, (h, w), mode='area')
        loss += inverse_depth_smoothness_loss(disp, img_scaled) * weight
        weight /= 2.3
    return loss

def spatial_normalize(disparities):
    '''
    proposed by https://arxiv.org/abs/1712.00175
    '''
    disp_mean = disparities.mean(1).mean(1).mean(1)
    disp_mean = disp_mean.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    return disparities/disp_mean

@torch.no_grad()
def compute_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    batch_size = gt.size(0)
    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]
