import numpy as np
import numpy
import torch
import shutil
from path import Path
import datetime
from collections import OrderedDict
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
import os

def json_out(dictionary, outpath, outname):
	with open(os.path.join(outpath, outname), 'w', encoding="utf-8") as f_out:
		jsObj = json.dumps(dictionary, indent=4)
		f_out.write(jsObj)
		f_out.close()

def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    keys_with_prefix['sequence_length'] = 'seq'
    keys_with_prefix['padding_mode'] = 'padding_'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'
    keys_with_prefix['photo_loss_weight'] = 'p'
    keys_with_prefix['smooth_loss_weight'] = 's'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    return save_path/timestamp

def save_checkpoint(save_path, dispnet_state, visualnet_state, imunet_state, posenet_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['UnVIO_dispnet','UnVIO_visualnet', 'UnVIO_imunet', 'UnVIO_posenet']
    states = [dispnet_state, visualnet_state, imunet_state, posenet_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}'.format(prefix,filename))

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix,filename), save_path/'{}_best.pth.tar'.format(prefix))

def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0,1,low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0,max_value,resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:,i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)

def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )
    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}


def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
    return array


def tensor2array2(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
    else:
        raise ValueError('wrong dimention of tensor.shape {}'.format(tensor.shape))
    return (array.transpose(1,2,0) * 255).astype('uint8')

def tensor2img(tensor):
    img = tensor.cpu().data.numpy()
    mean = np.array([0.5, 0.5, 0.5])
    img = np.transpose(img, (0, 2, 3, 1))
    img = img*mean + mean
    return img

def show_imgs(imgs):
    '''
    from utils import tensor2img as t2i
    from utils import show_imgs as shi
    tgi = t2i(tgt_img); rfi = t2i(ref_img); rwi = t2i(ref_img_warped); rci = t2i(rec_imgs[i])
    tmp = diff_mask.squeeze().cpu().data.numpy() ; tmp = occ_mask.squeeze().cpu().data.numpy()
    m=0; shi([tgi[m], rwi[m], rfi[m], rci[m], tmp[m]]); m=0; shi([tgi[m], rwi[m], rfi[m], tmp[m]])
    import matplotlib.pyplot as plt
    plt.subplot(311); plt.imshow(tgi[m]);plt.subplot(312); plt.imshow(rfi[m]); plt.subplot(313); plt.imshow(tgi[m]); plt.imshow(tmp[m],alpha=0.2,cmap='magma');plt.show()
    '''
    if type(imgs) not in [list]:
        assert((len(imgs.shape)==3 and imgs.shape[2]==3) or len(imgs.shape)==2)
        plt.imshow(imgs)
        plt.show()
    else:
        num = len(imgs)
        for i, img in enumerate(imgs):
            assert((len(img.shape)==3) or len(img.shape)==2)
            plt.subplot(num, 1, i+1)
            plt.imshow(img)
        plt.show()


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    """
    numpy.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = numpy.zeros((3, 3))
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = numpy.linalg.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity(3))
    if (numpy.linalg.det(U) * numpy.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U * S * Vh

    rotmodel = rot * model_zerocentered
    dots = 0.0
    norms = 0.0

    for column in range(data_zerocentered.shape[1]):
        dots += numpy.dot(data_zerocentered[:, column].transpose(), rotmodel[:, column])
        normi = numpy.linalg.norm(model_zerocentered[:, column])
        norms += normi * normi

    s = float(dots / norms)

    transGT = data.mean(1) - s * rot * model.mean(1)
    trans = data.mean(1) - rot * model.mean(1)

    model_alignedGT = s * rot * model + transGT
    model_aligned = rot * model + trans

    alignment_errorGT = model_alignedGT - data
    alignment_error = model_aligned - data

    trans_errorGT = numpy.sqrt(numpy.sum(numpy.multiply(alignment_errorGT, alignment_errorGT), 0)).A[0]
    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(alignment_error, alignment_error), 0)).A[0]

    return rot, transGT, trans_errorGT, trans, trans_error, s, np.array(data.transpose()), np.array(model_alignedGT.transpose())