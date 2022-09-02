import os
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from utils.logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
from utils.utils import save_path_formatter, save_checkpoint, tensor2array, json_out
from utils.pose_transfer import *
from utils import custom_transform
from losses.warp_loss_function import *
from utils.inverse_warp import *
import argparse
import csv
import subprocess

parser = argparse.ArgumentParser(description='UnVIO training on KITTI and Malaga Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset_root', type=str, required=True, help='dataset root path')
parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                    help='dataset format, stacked: stacked frames (from original TensorFlow code) \
                    sequential: sequential folders (easier to convert to with a non KITTI/Cityscape dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=5)
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                    help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--dataset', type=str, choices=['KITTI', 'Malaga'], default='KITTI',
                    help='which dataset is used to train, \'KITTI\' or \'Malaga')
parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--view-freq', default=200, type=int, dest='view_freq',
                    metavar='N', help='view frequency')
parser.add_argument('-e', '--debug', dest='debug', action='store_true',
                    help='debug =======')
parser.add_argument('--pretrained-dispnet', dest='pretrained_dispnet', default=None, metavar='PATH',
                    help='path to pre-trained DispNet')
parser.add_argument('--pretrained-imunet', dest='pretrained_imunet', default=None, metavar='PATH',
                    help='path to pre-trained IMUNet')
parser.add_argument('--pretrained-posenet', dest='pretrained_posenet', default=None, metavar='PATH',
                    help='path to pre-trained PoseNet')
parser.add_argument('--pretrained-visualnet', dest='pretrained_visualnet', default=None, metavar='PATH',
                    help='path to pre-trained VisualNet')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-m', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss', metavar='W', default=0)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('-d', '--td-loss-weight', type=float, help='weight for 3d loss', metavar='W', default=0.1)
parser.add_argument('--using-sliding-window', default=True, action='store_true', help='using sliding window optimization')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

n_iter = 0
best_error = -1
start_epoch = 0
args = parser.parse_args()  

def main():
    global best_error, n_iter, device, start_epoch
    data_root = args.dataset_root
    if args.using_sliding_window:
        print('=> warning: using sliding window optimization')
    else:
        print('=> warning: not using sliding window optimization')
    torch.manual_seed(args.seed)
    args.imu_range = [-10, 0]
    if args.dataset == 'KITTI':
        disp_alpha, disp_beta = 10, 0.01
        from dataset.KITTIDataset import DataSequence as dataset
        args.data = '{}/KITTI_rec_256'.format(data_root)
        args.img_width = 832
        args.img_height = 256
    elif args.dataset == 'Malaga':
        disp_alpha, disp_beta = 10, 0.01
        from dataset.MalagaDataset import DataSequence as dataset
        args.data = '{}/Malaga_Down/'.format(data_root)
        args.img_width = 832
        args.img_height = 256

    save_path = save_path_formatter(args, parser)
    args.save_path = '{}/UnVIO_saved_models/'.format(os.path.dirname(data_root))/save_path
    if not args.debug:
        print('=> will save everything to {}'.format(args.save_path))
        args.save_path.makedirs_p()
        train_writer = SummaryWriter(args.save_path)   
    else:
        train_writer = None 
    json_out(vars(args), args.save_path, 'config.json')

    train_transform = custom_transform.Compose([
        custom_transform.ToTensor(),
        custom_transform.AugmentImagePair(),
        custom_transform.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
    ])

    val_transform = custom_transform.Compose([
        custom_transform.ToTensor(),
        custom_transform.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
    ])

    train_set = dataset(root=args.data, 
                        seed=0, 
                        train=True, 
                        sequence_length=args.sequence_length,
                        imu_range=args.imu_range,
                        transform=train_transform,
                        image_width=args.img_width,
                        image_height=args.img_height)
    if args.dataset == 'KITTI':
        val_set = dataset(root=args.data, 
                        seed=0, 
                        train=False, 
                        sequence_length=args.sequence_length,
                        imu_range=args.imu_range,
                        transform=val_transform,
                        image_width=args.img_width,
                        image_height=args.img_height,
                        shuffle=False,
                        scene=['2011_09_30_drive_0033_sync_02', '2011_09_30_drive_0034_sync_02'])
    else:
        val_set = dataset(root=args.data, 
                        seed=0, 
                        train=False, 
                        sequence_length=args.sequence_length,
                        imu_range=args.imu_range,
                        transform=val_transform,
                        image_width=args.img_width,
                        image_height=args.img_height,
                        shuffle=False)
    
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)


    # disp_net, visual_net, imu_net, pose_net
    disp_net = models.DepthDecoder(alpha=disp_alpha, beta=disp_beta).to(device)
    visual_net = models.models.VisualNet().to(device)
    imu_net = models.models.ImuNet().to(device)
    pose_net = models.models.PoseNet(input_size=1024).to(device)

    if args.pretrained_visualnet:
        print("=> using pre-trained weights for Visualnet")
        weights = torch.load(args.pretrained_visualnet)
        visual_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        visual_net.init_weights()

    if args.pretrained_dispnet:
        print("=> using pre-trained weights for Dispnet")
        weights = torch.load(args.pretrained_dispnet)
        disp_net.load_state_dict(weights['state_dict'])
    else:
        disp_net.init_weights()

    if args.pretrained_imunet:
        print("=> using pre-trained weights for IMUnet")
        weights = torch.load(args.pretrained_imunet)
        imu_net.load_state_dict(weights['state_dict'])
    else:
        imu_net.init_weights()

    if args.pretrained_posenet:
        print("=> using pre-trained weights for Posenet")
        weights = torch.load(args.pretrained_posenet)
        pose_net.load_state_dict(weights['state_dict'])
    else:
        pose_net.init_weights()

    cudnn.benchmark = True
    disp_net = torch.nn.DataParallel(disp_net)
    visual_net = torch.nn.DataParallel(visual_net)
    imu_net = torch.nn.DataParallel(imu_net)
    pose_net = torch.nn.DataParallel(pose_net)

    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr},
        {'params': visual_net.parameters(), 'lr': args.lr},
        {'params': imu_net.parameters(), 'lr': args.lr},
        {'params': pose_net.parameters(), 'lr': args.lr}
    ]
    
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()
    if args.dataset == 'KITTI':
        with open(args.save_path/'valid_result.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['09', '', '10', ''])
            writer.writerow(['tl', 'rl', 'tl', 'rl'])
    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)     
        logger.reset_train_bar()
        # train
        train_loss = train(args, train_loader, disp_net, visual_net, imu_net, pose_net, optimizer, train_writer, logger, epoch)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # validation
        logger.reset_valid_bar()
        if args.dataset == 'Malaga':
            val_loss, _ = validate(args, val_loader, disp_net, visual_net, imu_net, pose_net, train_writer, logger, epoch)
            temp_error = val_loss
            train_writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': temp_error}, epoch)
            if best_error < 0:
                best_error = temp_error
            is_best = temp_error <= best_error
            best_error = min(best_error, temp_error)

            logger.valid_writer.write('* Chkpt: temp_error {:.4f}, mini_error {:.4f}'.format(temp_error, best_error))
        elif args.dataset == 'KITTI':
            tl, rl = validate(args, val_loader, disp_net, visual_net, imu_net, pose_net, train_writer, logger, epoch)
            temp_error = np.mean(tl)
            train_writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': temp_error}, epoch)
            if best_error < 0:
                best_error = temp_error
                best_tl, best_rl = tl, rl
            train_writer.add_scalars('eval_rl', {'09_rl': rl[0], '10_rl': rl[1]}, epoch)
            train_writer.add_scalars('eval_tl', {'09_tl': tl[0], '10_tl': tl[1]}, epoch)
            
            is_best = temp_error <= best_error
            best_error = min(best_error, temp_error)
            if is_best:
                best_tl, best_rl = tl, rl
            logger.valid_writer.write('* Chkpt: 09_rl {:.4f}, 09_tl {:.4f}, 10_rl {:.4f}, 10_tl {:.4f} \n   Best: 09_rl {:.4f}, 09_tl {:.4f}, 10_rl {:.4f}, 10_tl {:.4f}'.format(
                                                        rl[0], tl[0], rl[1], tl[1], best_rl[0], best_tl[0], best_rl[1], best_tl[1]))
        save_checkpoint(args.save_path,
        {
            'epoch': epoch+1,
            'state_dict': disp_net.module.state_dict()},
        {
            'epoch': epoch+1,
            'state_dict': visual_net.module.state_dict()
        },
        {
            'epoch': epoch+1,
            'state_dict': imu_net.module.state_dict()
        },
        {
            'epoch': epoch+1,
            'state_dict': pose_net.module.state_dict()
        }, is_best)

        if args.dataset == 'Malaga' and is_best:
            history_path = args.save_path + '/history/'
            folder = str(epoch).zfill(2)
            history_path = history_path + folder
            if not os.path.isdir(history_path):
                os.makedirs(history_path)

            MODEL_PATH= args.save_path
            POSENET_FEA_MODEL='/UnVIO_visualnet_best.pth.tar'
            DEEPVIO_IMU_MODEL='/UnVIO_imunet_best.pth.tar'
            DEEPVIO_POSE_MODEL='/UnVIO_posenet_best.pth.tar'
            test_stdout = os.path.join(MODEL_PATH, 'Malaga_out.txt')
            with open(test_stdout, 'a') as stdout:
                for j in ['05','06','08']:
                    cmd = 'python test_pose.py \
                    --pretrained-visualnet {} \
                    --pretrained-imunet {} \
                    --pretrained-posenet {} \
                    --dataset_root {} \
                    --dataset Malaga \
                    --testscene {}'.format(MODEL_PATH+POSENET_FEA_MODEL, MODEL_PATH + DEEPVIO_IMU_MODEL, MODEL_PATH + DEEPVIO_POSE_MODEL, data_root, j)
                    p = subprocess.Popen(cmd, shell=True, stdout=stdout)
                    p.wait()
                    os.system('mv {}/predpose_{}.csv {}/'.format(args.save_path, j, history_path))

    logger.epoch_bar.finish()


def save_trajectory(root, absolute_pose, name, epoch):
    history_path = root + '/history/'
    folder = str(epoch).zfill(2)
    history_path = os.path.join(history_path, folder)
    if not os.path.isdir(history_path):
        os.makedirs(history_path)
    np.savetxt(history_path/'{}.txt'.format(name), absolute_pose.reshape(-1, 12))

def train(args, train_loader, disp_net, visual_net, imu_net, pose_net, optimizer, train_writer, logger, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    end = time.time()
    global n_iter, device
    alpha1, alpha2, alpha3, alpha4 = args.photo_loss_weight, 0.1, 0.1, args.smooth_loss_weight

    disp_net.train()
    visual_net.train()
    imu_net.train()
    pose_net.train()

    using_sliding_window = args.using_sliding_window
    scales = 4
    for i, (imgs, imus, intr, gt) in enumerate(train_loader):
        data_time.update(time.time() - end)
        imgs = [img.to(device) for img in imgs]      # B S 3 H W 
        imus = imus.to(device)      # B S T 6 
        intr = intr.to(device)
        gt = gt.to(device)
        if using_sliding_window:
            if args.sequence_length == 5:
                Input = torch.cat(imgs[1:4], dim=0)
                disp = disp_net(Input)
                bat_s = disp[0].shape[0] // 3
                depth = [1/dis for dis in disp]
                depth1 = [d[args.batch_size*0: args.batch_size*1] for d in depth]
                depth2 = [d[args.batch_size*1: args.batch_size*2] for d in depth]
                depth3 = [d[args.batch_size*2: args.batch_size*3] for d in depth]
                visual_feature = visual_net(imgs)               # B 4 512
                imu_feature = imu_net(imus[:, 1:])              # B, 4, 512
                out = pose_net(visual_feature, imu_feature)     # B, 4, 6
                out_w_pose = []
                for j in range(3):
                    tmp_out = pose_net(visual_feature[:, j:j+2], imu_feature[:, j:j+2])   # B, 2, 6
                    out_w_pose.append(tmp_out)

                out_w_pose_avg = [out_w_pose[0][:, 0], (out_w_pose[0][:, 1]+out_w_pose[1][:, 0])/2,
                                 (out_w_pose[1][:, 1]+out_w_pose[2][:, 0])/2, out_w_pose[2][:, 1]]    # T12 (T23) (T34) T45
                out_w_pose_avg = torch.stack(out_w_pose_avg, dim=1)  # B, 4, 6
            elif args.sequence_length == 7:
                middle_index = [1, 3, 5]
                Input = torch.cat([imgs[j] for j in middle_index], dim=0)
                disp = disp_net(Input)
                depth = [1/dis for dis in disp]
                visual_feature = visual_net(imgs)               # B, seq-1, 512
                imu_feature = imu_net(imus[:,1:])               # B, seq-1, 512
                out = pose_net(visual_feature, imu_feature)     # B, seq-1, 6

                out_w_pose = []
                for j in middle_index:
                    tmp_out = pose_net(visual_feature[:, j-1:j+1], imu_feature[:, j-1:j+1])     # B, 2, 6
                    out_w_pose.append(tmp_out)
                out_w_pose_avg = torch.cat(out_w_pose, dim=1)

        else:
            Input = imgs[args.sequence_length//2]
            disp = disp_net(Input)
            depth = [1/dis for dis in disp]
            depth2 = depth

            visual_feature = visual_net(imgs)               # B S 1024
            imu_feature = imu_net(imus[:,1:])               # B T 1000
            out = pose_net(visual_feature, imu_feature)     # B T-1 6

        if using_sliding_window:
            if args.sequence_length == 5:
                loss_photo, loss_3d = 0, 0
                for j in range(3):
                    pose_j = out2posew(out_w_pose[j])
                    depth_j = [d[args.batch_size*j: args.batch_size*(j+1)] for d in depth]
                    if j != 1:
                        tmp1, tmp2 = photometric_reconstruction_loss(imgs[j+1], imgs[j:j+1]+imgs[j+2:j+3], intr,
                                                         depth_j[:scales], pose_j, args.rotation_mode, args.padding_mode, ref_depth=[None]*scales)
                    else:
                        tmp1, tmp2 = photometric_reconstruction_loss(imgs[j+1], imgs[j:j+1]+imgs[j+2:j+3], intr,
                                                         depth_j[:scales], pose_j, args.rotation_mode, args.padding_mode, ref_depth=[[depth1[s], depth3[s]] for s in range(scales)])

                    loss_photo += tmp1
                    loss_3d += tmp2
                pose = out2pose(out, args)
                loss_vo1 = voloss(out_w_pose[0][:, 1], out_w_pose[1][:, 0]) + voloss(out_w_pose[1][:, 1], out_w_pose[2][:, 0])
                loss_vo2 = voloss(out, out_w_pose_avg)
            elif args.sequence_length == 7:
                loss_photo, loss_3d = 0, 0
                for j, idx in enumerate(middle_index):  
                    pose_j = out2posew(out_w_pose[j])
                    depth_j = [d[args.batch_size*j : args.batch_size*(j+1)] for d in depth]
                    tmp1, tmp2 = photometric_reconstruction_loss(imgs[idx], imgs[idx-1:idx]+imgs[idx+1:idx+2], intr,
                                                         depth_j[:scales], pose_j, args.rotation_mode, args.padding_mode, ref_depth=[None]*scales)
                    loss_photo += tmp1
                    loss_3d += tmp2

                pose = out2pose(out[:,1:5], args)
                idx = (args.sequence_length-1)//2
                depth2 = [d[args.batch_size : args.batch_size*2] for d in depth]
                tmp1, tmp2 = photometric_reconstruction_loss(imgs[idx], imgs[idx-2:idx-1]+imgs[idx+2:idx+3], intr,
                                                         depth2[:scales], pose[:, 0:4:3], args.rotation_mode,
                                                         args.padding_mode, ref_depth=[[d[:args.batch_size], d[args.batch_size*2:]] for d in depth])
                loss_photo += tmp1
                loss_3d += tmp2

                loss_vo1 = 0
                loss_vo2 = voloss(out, out_w_pose_avg)

        else:   
            pose = out2pose(out, args)
            loss_photo, loss_3d= photometric_reconstruction_loss(imgs[args.sequence_length//2], imgs[:args.sequence_length//2]+imgs[(args.sequence_length//2+1):], intr,
                                                     depth2[:scales], pose, args.rotation_mode, args.padding_mode, ref_depth=[None]*scales)
            loss_vo1 = 0
            loss_vo2 = 0

        if args.dataset == 'Malaga':
            loss_smooth = disp_smooth_loss(depth[:scales], Input)
        elif args.dataset == 'KITTI' and not using_sliding_window:
            loss_smooth = disp_smooth_loss(depth[:scales], Input) 
        else:
            loss_smooth = disp_smooth_loss(depth[:scales], Input)
            # loss_smooth = disp_smooth_loss(disp[:scales], Input)

        loss = alpha1*loss_photo + alpha2*loss_vo1 + alpha3*loss_vo2 + alpha4*loss_smooth + alpha2*loss_3d

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), args.batch_size)

        n_iter += 1
        batch_time.update(time.time() - end)
        end = time.time()
        logger.train_bar.update(i+1)
        if i % 3 == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {} Epoch {}'.format(batch_time, data_time, losses, epoch))
        if i >= args.epoch_size - 1:
            break

        ###########################  train_writer ################################
        if not args.debug:
            train_writer.add_scalar('photometric_error', loss_photo.item(), n_iter)
            train_writer.add_scalar('disparity_smoothness_loss', loss_smooth.item(), n_iter)
            if args.using_sliding_window:
                if args.sequence_length == 5:
                    train_writer.add_scalar('vo1_loss', loss_vo1.item(), n_iter)
                train_writer.add_scalar('vo2_loss', loss_vo2.item(), n_iter)
                train_writer.add_scalar('3d_loss', loss_3d.item(), n_iter)
            train_writer.add_scalar('total_loss', loss.item(), n_iter)
            if args.view_freq > 0 and n_iter % args.view_freq == 0 and not (args.using_sliding_window and args.sequence_length==7):
                tgt_img = imgs[args.sequence_length//2]
                if args.using_sliding_window and args.sequence_length == 7:
                    ref_imgs = imgs[1:args.sequence_length//2] + imgs[(args.sequence_length//2+1):-1]
                else:
                    ref_imgs = imgs[:args.sequence_length//2] + imgs[(args.sequence_length//2+1):]
                train_writer.add_image('train Input', tensor2array(tgt_img[0]), n_iter)

                with torch.no_grad():
                    for k, scaled_depth in enumerate(depth2[:1]):
                        train_writer.add_image('train Dispnet Output Normalized {}'.format(k),
                                               tensor2array(1/scaled_depth[0], max_value=None, colormap='magma'),
                                               n_iter)
                        train_writer.add_image('train Depth Output Normalized {}'.format(k),
                                               tensor2array(scaled_depth[0], max_value=None),
                                               n_iter)
                        b, _, h, w = scaled_depth.size()
                        downscale = tgt_img.size(2)/h

                        tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
                        ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]

                        intrinsics_scaled = torch.cat((intr[:, 0:2]/downscale, intr[:, 2:]), dim=1)

                        # log warped images along with explainability mask
                        for j, ref in enumerate(ref_imgs_scaled):
                            ref_warped = inverse_warp(ref, scaled_depth[:, 0], pose[:, j],
                                                      intrinsics_scaled,
                                                      rotation_mode=args.rotation_mode,
                                                      padding_mode=args.padding_mode, 
                                                      )[0]
                            train_writer.add_image('train Warped Outputs {} {}'.format(k, j),
                                                   tensor2array(ref_warped),
                                                   n_iter)
                            train_writer.add_image('train ref_img {} {}'.format(k, j),
                                                   tensor2array(ref[0]),
                                                   n_iter)

    return losses.avg[0]

def voloss(pose1, pose2, k=10):
    if pose1.shape[1] == 6:
        rot1, tra1 = pose1[:, :3], pose1[:, 3:6]
        rot2, tra2 = pose2[:, :3], pose2[:, 3:6]
    elif pose1.shape[2] == 6:
        rot1, tra1 = pose1[:, :, :3], pose1[:, :, 3:6]
        rot2, tra2 = pose2[:, :, :3], pose2[:, :, 3:6]
    return k*(rot1 - rot2).abs().mean() + (tra1 - tra2).abs().mean()

@torch.no_grad()
def validate(args, val_loader, disp_net, visual_net, imu_net, pose_net, train_writer, logger, epoch=0):
    global device
    alpha1, alpha2, alpha3, alpha4 = 1, 0.1, 0.05, 0.1

    batch_time = AverageMeter()
    losses = AverageMeter(precision=5)
    disp_net.eval()
    visual_net.eval()
    imu_net.eval()
    pose_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    predictions_array = []
    if args.dataset == 'KITTI':
        cpp = './eval/devkit/cpp/evaluate_odometry'
        gt_path = './eval/gt/poses'
        seq = '[09,10]'
        GT = [np.loadtxt(gt_path+'/{}.txt'.format(s)).reshape(-1, 3, 4) for s in ['09', '10']]

    item = 0
    off_set = 5 - args.sequence_length
    for i, (imgs, imus, intr, gt) in enumerate(val_loader):

        imgs = [img.to(device) for img in imgs]      # B S T 6 
        tgt_img = imgs[args.sequence_length//2]
        ref_imgs = imgs[:args.sequence_length//2] + imgs[(args.sequence_length//2+1):]
        imus = imus.to(device)      # B S T 6 
        intr = intr.to(device)
        gt = gt.to(device)

        out1 = disp_net(tgt_img)
        disp = [out1]
        depth = [1/dis for dis in disp]

        visual_feature = visual_net(imgs)           # B 4 512
        imu_feature = imu_net(imus[:,1:])        # B, 4, 512
        if args.dataset == 'Malaga':
            out = pose_net(visual_feature, imu_feature)
        elif args.dataset == 'KITTI':
            out = pose_net(visual_feature, imu_feature).data.cpu().numpy()
        if args.dataset == 'Malaga':
            pose = out2pose(out, args)
            loss_photo = photometric_reconstruction_loss(tgt_img, ref_imgs, intr,
                                                     depth[:1], pose, args.rotation_mode, args.padding_mode, ref_depth=[None])[0].item()
            loss = alpha1*loss_photo
            losses.update(loss, args.batch_size)
        elif args.dataset == 'KITTI':
            for out_item in out:
                if item == 0 or item == 1590+off_set:
                    predictions_array.append(np.zeros([1, 6]))
                    for j in range(out_item.shape[0]):
                        predictions_array.append(out_item[j:, :])
                elif item == 1589+off_set:
                    predictions_array.append(out_item[-1:, :])
                    absolute_pose = np.array(relative2absolute(predictions_array))[:, :3]
                    scale = scale_lse_solver(absolute_pose[:, :, 3], GT[0][:, :, 3])
                    absolute_pose[:, :, 3] *= scale
                    np.savetxt(args.save_path/'09.txt', absolute_pose.reshape(-1, 12))
                    save_trajectory(args.save_path, absolute_pose, '09', epoch)
                    predictions_array = []
                elif item == 1590+off_set+1219+off_set:
                    predictions_array.append(out_item[-1:, :])
                    absolute_pose = np.array(relative2absolute(predictions_array))[:, :3]
                    scale = scale_lse_solver(absolute_pose[:, :, 3], GT[1][:, :, 3])
                    absolute_pose[:, :, 3] *= scale
                    np.savetxt(args.save_path/'10.txt', absolute_pose.reshape(-1, 12))
                    save_trajectory(args.save_path, absolute_pose, '10', epoch)
                else:
                    predictions_array.append(out_item[-1:, :])
                item += 1
        batch_time.update(time.time() - end)
        end = time.time()
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % 3 == 0:
            if args.dataset == 'Malaga':
                logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))
    logger.valid_bar.update(len(val_loader))
    if args.dataset == 'Malaga':
        return losses.avg[0], 0
    elif args.dataset == 'KITTI':
        test_stdout = os.path.join(args.save_path, 'KITTI_out.txt')
        test_errout = os.path.join(args.save_path, 'KITTI_error_out.txt')
        with open(test_stdout, 'a') as stdout, open(test_errout, 'a') as errout:
            error_dir = args.save_path/'errors'
            plot_path_dir = args.save_path/'plot_path'
            plot_error_dir = args.save_path/'plot_error'
            error_dir.makedirs_p(); plot_path_dir.makedirs_p(); plot_error_dir.makedirs_p()
            cmd = '{} {} {} {}'.format(cpp, gt_path, str(args.save_path), seq)
            p = subprocess.Popen(cmd, shell=True, stdout=stdout, stderr=errout)
            p.wait()
            rl, tl = [], []
            for item in ['09', '10']:
                rl.append(get_rl(args.save_path/'plot_error/{}_rl.txt'.format(item)))
                tl.append(get_tl(args.save_path/'plot_error/{}_tl.txt'.format(item))) 
            return tl, rl

def get_rl(path):
    rl = np.loadtxt(path)
    rl = np.mean(rl[:, 1])*100*57.3
    return rl

def get_tl(path):
    tl = np.loadtxt(path)
    tl = np.mean(tl[:, 1])*100
    return tl

def relative2absolute(pose):
    abs_pose_mat = []
    for i in range(len(pose)):
        temp_mat = _6Dofto16mat(pose[i])
        if i == 0:
            abs_pose_mat.append(temp_mat)
        else:
            abs_pose_mat.append(abs_pose_mat[i-1] @ temp_mat)
    return abs_pose_mat

def _6Dofto16mat(pose):
    translation = pose[0][3:]
    rotation = pose[0][:3]
    R = euler_matrix(rotation[0], rotation[1], rotation[2])
    T = np.vstack([np.hstack([R, translation.reshape(-1, 1)]), [0, 0, 0, 1]])
    return T

def out2pose(out, args):
    pose = [pose_vec2mat4(out[:, i]) for i in range(args.sequence_length-1)]
    if len(pose) == 4:
        pose = [pose[0] @ pose[1], pose[1], b_inv(pose[2]), b_inv(pose[2] @ pose[3])]
    elif len(pose) == 2:
        pose = [pose[0], b_inv(pose[1])]
    elif len(pose) == 6:
        pose = [pose[0] @ pose[1] @ pose[2], pose[1] @ pose[2], pose[2], b_inv(pose[3]), b_inv(pose[3] @ pose[4]), b_inv(pose[3] @ pose[4] @ pose[5])]
    pose = torch.stack(pose, dim=1)
    return pose[:, :, :3, :]

def out2posew(out):
    seq_len = out.shape[1]
    pose = [pose_vec2mat4(out[:, i]) for i in range(seq_len)]
    pose = pose[:seq_len//2] + [b_inv(p) for p in pose[seq_len//2:]]
    pose = torch.stack(pose, dim=1)
    return pose[:, :, :3, :]

def scale_lse_solver(X, Y):
    """Least-sqaure-error solver
    Compute optimal scaling factor so that s(X)-Y is minimum
    Args:
        X (KxN array): current data
        Y (KxN array): reference data
    Returns:
        scale (float): scaling factor
    """
    scale = np.sum(X * Y)/np.sum(X ** 2)
    return scale

if __name__ == "__main__":
    main()