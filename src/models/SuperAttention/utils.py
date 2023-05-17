import numpy as np
import skimage.color
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.util import img_as_float
from skimage.segmentation import slic
from torchvision.transforms import Grayscale
from skimage.segmentation import mark_boundaries

import shutil
import os
import torch.distributed as dist
import random
from skimage import color, io


def totalv_loss(img):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :h_img-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :w_img-1], 2).sum()
    return 2 * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

def save_ckp(state, is_best, checkpoint_dir, idx ,best_model_dir):
    f_path = checkpoint_dir + '/checkpoint_'+str(idx)+'.pt'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, f_path)
    if is_best:
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        best_fpath = best_model_dir + '/best_model_'+str(idx)+'.pt'
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def img_segments_only(img_grays, div_resize, num_max_seg):
    """
    :param img_grays: Target image
    :param div_resize: Resizing factor
    :return: Superpixel's label map for the target and reference images.
    """

    # Gray scale or RGB image
    if img_grays.ndim == 2:
        image_gray_2 = img_as_float(resize(img_grays, (img_grays[0].size / div_resize, img_grays[1].size / div_resize)))
        segment_gray_2 = slic(image_gray_2, n_segments=int(num_max_seg), channel_axis=None)

    else:
        image_gray_2 = img_as_float(resize(img_grays, (len(img_grays[0]) / div_resize, len(img_grays[1]) / div_resize)))
        segment_gray_2 = slic(image_gray_2, n_segments=int(num_max_seg), sigma=0, compactness=9)


    return segment_gray_2

def ref_to_target(image_gray, image_ref, indx_similarity, segment_gray, segments_ref):
    """
    :param image_gray: Target image
    :param image_ref: Reference image
    :param indx_similarity: indices in which the similarity is maximum per row
    :param segment_gray: Superpixel segmented mask for the target image
    :param segments_ref: Superpixel segmented mask for the reference image
    :return: image_colored: Target colored image by copying the average superpixel values from the reference image.
    """

    # Gray scale or RGB image
    if image_gray.ndim == 2:
        replicate_target = np.repeat(image_gray[:, :, np.newaxis], 3, axis=2)
    else:
        replicate_target = image_gray


    image_colored = np.zeros(replicate_target.shape) # initializing 3D array which will be the final colored image

    for i in range(len(indx_similarity)):            # Running over the indices (Normally it's the same as segmented target len)
        mask = (segments_ref == indx_similarity[i])  # Running over every reference superpixel mask that match one from the target
        indx_non_zero = np.nonzero(mask)                # Extracting the mask.
        img_ref_mask_mean = np.mean(image_ref[indx_non_zero[0], indx_non_zero[1], :], axis=0) # Calculating mean column wise (channels)
        mask_target = (segment_gray == i) # Mask in target image
        indx_non_zero_target = np.nonzero(mask_target)
        image_colored[indx_non_zero_target[0], indx_non_zero_target[1], :] = img_ref_mask_mean #Copy mean value in corresponding segmented pixels
    return image_colored


def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

def plot_text_label_wt_bd(segment, image):
    regions = regionprops(segment)
    p = 0
    cx = np.zeros(np.max(segment))
    cy = np.zeros(np.max(segment))
    v = np.zeros(np.max(segment))
    for props in regions:
        cx[p], cy[p] = props.centroid  # centroid coordinates
        v[p] = props.label  # value of label
        p = p + 1

    for j in range(np.max(segment)):
        plt.imshow(image)
        plt.text(cy[j], cx[j], '{}'.format(int(v[j])), fontsize=5, color='r')
        if j == (np.max(segment) - 1):
            break

    plt.axis("off")

def plot_text_label(segment, image):
    regions = regionprops(segment)
    p = 0
    cx = np.zeros(np.max(segment))
    cy = np.zeros(np.max(segment))
    v = np.zeros(np.max(segment))
    for props in regions:
        cx[p], cy[p] = props.centroid  # centroid coordinates
        v[p] = props.label  # value of label
        p = p + 1

    for j in range(np.max(segment)):
        plt.imshow(mark_boundaries(image, segment))
        plt.text(cy[j], cx[j], '{}'.format(v[j]), fontsize=5, color='r')
        if j == (np.max(segment) - 1):
            break

    plt.axis("off")


def pause():
    programPause = input("Press the <ENTER> key to continue...")

def plot_colored_img(indx_similarity, segment, image_colored):

    regions = regionprops(segment)
    p = 0
    cx = np.zeros(np.max(segment))
    cy = np.zeros(np.max(segment))
    v = np.zeros(np.max(segment))
    for props in regions:
        cx[p], cy[p] = props.centroid  # centroid coordinates
        v[p] = props.label  # value of label
        p = p + 1

    for j in range(len(indx_similarity)-1):
        plt.imshow(image_colored)
        plt.text(cy[j], cx[j], '{}'.format(indx_similarity[j + 1]), fontsize=5)
        if j == (len(indx_similarity)):
            break

    plt.axis("off")


def plot_features(features, nb_features, min_v, max_v):
    fig3 = plt.figure()
    # plot 64 feature map
    square = np.sqrt(nb_features).astype('int')

    # v_min = np.min(features)
    # v_max = np.max(features)
    v_min = min_v
    v_max = max_v
    for feat in (features):
        ix = 1
        for _ in range(square):
            for _ in range(square):
                ax8 = plt.subplot(square, square, ix)
                ax8.set_xticks([])
                ax8.set_yticks([])
                plt.imshow(features[ix - 1, :, :], vmin=v_min, vmax=v_max)
                plt.title(ix, y=0.94, fontsize=8)
                plt.colorbar()
                ix += 1
        if ix == nb_features+1:
            break

def plot_interactive(image_ref, image_target, segments_ref, segments_target, indx_sim, div ):

    image_ref = resize(image_ref, (int(224/div), int(224/div)))
    image_target = resize(image_target, (int(224 / div), int(224 / div)))

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(mark_boundaries(image_target, segments_target), cmap='gray')
    ax[1].imshow(mark_boundaries(image_ref, segments_ref))


    def on_press(event):
        cord_x, cord_y = int(event.xdata), int(event.ydata)
        label_target = segments_target[cord_y, cord_x]
        mask = (segments_target == label_target).astype(float)
        mask[mask == 0] = 0.6

        label_refs = np.where(indx_sim[label_target] > 0.9)[0]
        mask_ref = np.zeros_like(segments_ref.astype(float))
        for label_ref in label_refs:
            mask_ref[segments_ref == label_ref] = 1.
        mask_ref[mask_ref == 0] = 0.2

        ax[0].imshow(image_target * mask, cmap='gray')
        ax[1].imshow(image_ref * np.repeat(mask_ref[:,:,np.newaxis], 3 ,axis=2)) #np.repeat(mask_ref[:,:,np.newaxis], 3 ,axis=2))

        fig.canvas.draw()
        print('you pressed', event.button, cord_x, cord_y )


    cid = fig.canvas.mpl_connect('button_press_event', on_press)

    ax[0].axis('off')
    ax[1].axis('off')
    plt.show()

def count_parameters(model):

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def unnormalize_colorspace(lab_tensor, color_space='lab'):
    """
    :param: lab_tensor: Tensor in lab color space in range [0,1]

    :return: lab_tensor: Tensor in Lab color space in range L:[0,100], a and b [-128,127]
    """
    if color_space == 'lab':
        if len(lab_tensor[0, :, :, :]) == 2:
            lab_tensor[:, :, :, :] = (lab_tensor[:, :, :, :] * 255.0) - 128.0
        else:
            lab_tensor[:, 0, :, :] = lab_tensor[:, 0, :, :] * 100.0
            lab_tensor[:, 1:, :, :] = (lab_tensor[:, 1:, :, :] * 255.0) - 128.0

    if color_space == 'yuv':
        if len(lab_tensor[0, :, :, :]) == 2:
            lab_tensor[:, :, :, :] = lab_tensor[:, :, :, :] - 0.5
        else:
            lab_tensor[:, 1:, :, :] = lab_tensor[:, 1:, :, :] - 0.5
    return lab_tensor

def torch_colorspace_to_rgb(lab_tensor, color_space='lab'):
    # lab_tensor = torch.cat((l_tensor[:, 0, :, :].unsqueeze(1), ab_tensor), dim=1)
    rgb_pred = np.zeros(((len(lab_tensor)), (len(lab_tensor[0, 0, :, :])), (len(lab_tensor[0, 0, :, :])), (len(lab_tensor[0, :, :, :]))))
    lab_tensor_unorm = unnormalize_colorspace(lab_tensor, color_space)
    lab_tensor_unorm_torch = lab_tensor_unorm.permute(0, 2, 3, 1)
    lab_tensor_unorm = lab_tensor_unorm.detach().cpu().numpy()
    if color_space == 'lab':
        for i in range(len(lab_tensor_unorm)):
            numpy_img = lab_tensor_unorm[i, :, :, :].transpose((1, 2, 0))
            rgb_pred[i, :, :, :] = skimage.color.lab2rgb(numpy_img)
    if color_space == 'yuv':
        for i in range(len(lab_tensor_unorm)):
            numpy_img = lab_tensor_unorm[i, :, :, :].transpose((1, 2, 0))
            np.save('numpy_pred.npy', numpy_img)

            rgb_pred[i, :, :, :] = skimage.color.yuv2rgb(numpy_img)

    rgb_pred[rgb_pred > 1] = 1
    rgb_pred[rgb_pred < 0] = 0

    return rgb_pred, lab_tensor_unorm_torch

def unnormalize_colorspace_zhang(lab_tensor, color_space='lab'):
    """
    :param: lab_tensor: Tensor in lab color space in range [0,1]

    :return: lab_tensor: Tensor in Lab color space in range L:[0,100], a and b [-128,127]
    """
    if color_space == 'lab':
        if len(lab_tensor[0, :, :, :]) == 2:
            lab_tensor[:, :, :, :] = (lab_tensor[:, :, :, :] * 128.0)
        else:
            # lab_tensor[:, 0, :, :] = (lab_tensor[:, 0, :, :] * 100.0) + 50.0
            lab_tensor[:, 1:, :, :] = (lab_tensor[:, 1:, :, :] * 128.0)

    # if color_space == 'yuv':
    #     if len(lab_tensor[0, :, :, :]) == 2:
    #         lab_tensor[:, :, :, :] = lab_tensor[:, :, :, :] - 0.5
    #     else:
    #         lab_tensor[:, 1:, :, :] = lab_tensor[:, 1:, :, :] - 0.5
    return lab_tensor

def torch_colorspace_to_zhang(lab_tensor, color_space='lab'):
    rgb_pred = np.zeros(((len(lab_tensor)), (len(lab_tensor[0, 0, :, :])), (len(lab_tensor[0, 0, :, :])), (len(lab_tensor[0, :, :, :]))))
    lab_tensor_unorm = unnormalize_colorspace(lab_tensor, color_space)
    lab_tensor_unorm_torch = lab_tensor_unorm.permute(0, 2, 3, 1)
    lab_tensor_unorm = lab_tensor_unorm.detach().cpu().numpy()
    if color_space == 'lab':
        for i in range(len(lab_tensor_unorm)):
            numpy_img = lab_tensor_unorm[i, :, :, :].transpose((1, 2, 0))
            rgb_pred[i, :, :, :] = skimage.color.lab2rgb(numpy_img)
    return rgb_pred, lab_tensor_unorm_torch

def imagenet_norm(input, device):
    # VGG19 normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device=device, dtype=torch.float)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device=device, dtype=torch.float)
    out = (input - mean.view((1, 3, 1, 1))) / std.view((1, 3, 1, 1))
    return out

def img_torch(sample):
    img = sample
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)

    return img

def rand_ref(index_target, list_dataset, slic_target, labels, dist_scores, device, training = False , validation = False):
    thresh_category = 5
    if training == True:
        top_images = 10

    if validation == True:
        top_images = 1

    batch = len(index_target)
    index_target = index_target.int().numpy()
    labels_np = np.array(labels)
    _, h, w = slic_target[0].shape
    _, h_2, w_2 = slic_target[1].shape
    _, h_3, w_3 = slic_target[2].shape
    ref_out = torch.zeros((batch, 3, h, w))
    luminance_replicate_ref_out = torch.zeros((batch, 3, h, w))
    labels_torch = torch.zeros((batch, 1, h, w))
    labels_torch_2 = torch.zeros((batch, 1, h_2, w_2))
    labels_torch_3 = torch.zeros((batch, 1, h_3, w_3))
    label_masks_ref = []
    index_out = []


    # target_label = labels[index_target]

    labels_ref_all = slic_target[0]
    labels_ref_2_all = slic_target[1]
    labels_ref_3_all = slic_target[2]
    target_label = labels_np[index_target]
    # target_label = list(map(labels.__getitem__, index_target))

    scores_ref = dist_scores[index_target]

    for i in range(batch):
        best_indx = scores_ref[i]
        labels_best = labels_np[np.array(best_indx, dtype='int')]

        indx_corr_target = np.where(labels_best == target_label[i])[0]
        if indx_corr_target.size >= thresh_category:
            indx_proper_label = random.choice(indx_corr_target)
            ref_index = int(best_indx[indx_proper_label])
        else:
            indx_proper_label = random.choice(best_indx[0:top_images])
            ref_index = int(indx_proper_label)
            # ref_index = index_target[i]

        # print(index_target[i], ref_index, target_label[i], labels_best)

        # ref_index = random.choice(list_duplicates_of(labels, target_label[i]))
        ref = resize(io.imread('./'+list_dataset[int(ref_index)], pilmode='RGB'),
                     (256, 256))  # Reading Reference images in RGB
        ref_new_color = color.rgb2lab(ref)
        ref_luminance = ref_new_color[:, :, 0] / 100.0

        labels_ref = labels_ref_all[ref_index]
        labels_ref_2 = labels_ref_2_all[ref_index]
        labels_ref_3 = labels_ref_3_all[ref_index]


        ref_luminance = img_torch(ref_luminance[:, :, np.newaxis])
        luminance_replicate_ref = torch.cat((ref_luminance.float(), ref_luminance.float(), ref_luminance.float()),
                                            dim=0)
        ref_out[i, :, :, :] = img_torch(ref)
        luminance_replicate_ref_out[i, :, :, :] = luminance_replicate_ref

        labels_torch[i, :, :, :] = img_torch(labels_ref[:, :, np.newaxis]).type(torch.int64)
        labels_torch_2[i, :, :, :] = img_torch(labels_ref_2[:, :, np.newaxis]).type(torch.int64)
        labels_torch_3[i, :, :, :] = img_torch(labels_ref_3[:, :, np.newaxis]).type(torch.int64)

        index_out.append(ref_index)

    label_masks_ref.append(labels_torch.type(torch.int64))
    label_masks_ref.append(labels_torch_2.type(torch.int64))
    label_masks_ref.append(labels_torch_3.type(torch.int64))


    # labels_max = np.amax(labels_target_all)
    # labels_max_2 = np.max(labels_target_2_all)
    # labels_max_3 = np.amax(labels_target_3_all)

    return ref_out.to(device), luminance_replicate_ref_out.to(device), label_masks_ref, index_out

def calc_hist_classic(data_ab, device):
    N, C, H, W = data_ab.shape
    grid_a = torch.linspace(-1, 1, 21).view(1, 21, 1, 1, 1).expand(N, 21, 21, H, W).to(device)
    grid_b = torch.linspace(-1, 1, 21).view(1, 1, 21, 1, 1).expand(N, 21, 21, H, W).to(device)
    hist_a = torch.max(0.1 - torch.abs(grid_a - data_ab[:, 0, :, :].view(N, 1, 1, H, W)), torch.Tensor([0]).to(device)) * 10
    hist_b = torch.max(0.1 - torch.abs(grid_b - data_ab[:, 1, :, :].view(N, 1, 1, H, W)), torch.Tensor([0]).to(device)) * 10

    hist = (hist_a * hist_b).mean(dim=(3, 4)).view(N, -1).to(device)
    return hist


def calc_hist(data_ab, slic, alpha, device):
    N, C, H, W = data_ab.shape
    masks = torch.zeros(N, 1, H, W).to(device)
    for i in range(N):
        idx_target = torch.nonzero(alpha[i])[:, 0].to(device)
        masks[i] = torch.tensor([p.item() in idx_target for p in slic[i].reshape(-1)]).reshape(slic[i].shape)
    grid_a = torch.linspace(-1, 1, 21).view(1, 21, 1, 1, 1).expand(N, 21, 21, H, W).to(device)
    grid_b = torch.linspace(-1, 1, 21).view(1, 1, 21, 1, 1).expand(N, 21, 21, H, W).to(device)
    hist_a = torch.max(0.1 - torch.abs(grid_a - data_ab[:, 0, :, :].view(N, 1, 1, H, W)), torch.Tensor([0]).to(device)) * 10
    hist_b = torch.max(0.1 - torch.abs(grid_b - data_ab[:, 1, :, :].view(N, 1, 1, H, W)), torch.Tensor([0]).to(device)) * 10

    hist = (hist_a * hist_b * masks.unsqueeze(1).to(device)).mean(dim=(3, 4)).view(N, -1).to(device)
    # hist = (hist_a * hist_b).mean(dim=(3, 4)).view(N, -1).to(device)
    # io.imsave('./masks.png', masks[0][0].cpu().numpy())
    return hist

def calc_hist_ref(data_ab, slic, alpha, sim_mat, device):
    N, C, H, W = data_ab.shape
    masks = torch.zeros(N, 1, H, W).to(device)
    for i in range(N):
        idx_target = torch.nonzero(alpha[i])[:, 0].to(device)
        idx_ref = torch.unique(torch.nonzero(sim_mat[i][idx_target])[:, 1 ]).to(device)
        masks[i] = torch.tensor([p.item() in idx_ref for p in slic[i].reshape(-1)]).reshape(slic[i].shape)
    grid_a = torch.linspace(-1, 1, 21).view(1, 21, 1, 1, 1).expand(N, 21, 21, H, W).to(device)
    grid_b = torch.linspace(-1, 1, 21).view(1, 1, 21, 1, 1).expand(N, 21, 21, H, W).to(device)
    hist_a = torch.max(0.1 - torch.abs(grid_a - data_ab[:, 0, :, :].view(N, 1, 1, H, W)), torch.Tensor([0]).to(device)) * 10
    hist_b = torch.max(0.1 - torch.abs(grid_b - data_ab[:, 1, :, :].view(N, 1, 1, H, W)), torch.Tensor([0]).to(device)) * 10

    hist = (hist_a * hist_b * masks.unsqueeze(1).to(device)).mean(dim=(3, 4)).view(N, -1).to(device)
    # hist = (hist_a * hist_b).mean(dim=(3, 4)).view(N, -1).to(device)
    # io.imsave('./masks_ref.png', masks[0][0].cpu().numpy())
    return hist

def hist_loss(hist_t, hist_r):
    num = torch.square(hist_t - hist_r)
    den = hist_t + hist_r + 1e-5
    l_hist = 2 * torch.sum((num / den),  axis=1)
    l_hist = torch.mean(l_hist)
    return l_hist

def hist_inter(hist_t, hist_r):
    num = torch.sum(torch.min(hist_t, hist_r))
    den = torch.sum(hist_r) + 1e-5
    l_hist = num / den
    return l_hist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)
def save_checkpoint_multi(model,  checkpoint_dir):
    f_path = checkpoint_dir + '/checkpoint.pt'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    ckp = model.module.state_dict()
    torch.save(ckp, f_path)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def get_rank():

    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def is_main_process():

    return get_rank() == 0

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with torch.no_grad():
            dist.broadcast(p, 0)

from PIL import Image, ImageEnhance
import torch
from torchvision.transforms import ToPILImage,ToTensor


class ColorEnhance(object):
    def __init__(self, factor=1.5):
        self.factor=factor

    def __call__(self, x):
        if isinstance(x, Image.Image):
            return self._enhance_PIL(x)
        elif isinstance(x, torch.Tensor):
            if len(x.shape) == 4:
                xs = [self._enhance_Tensor(each)[None, ...] for each in x]
                x = torch.cat(xs)
                return x

            elif len(x.shape) == 3:
                return self._enhance_Tensor(x)
            else:
                raise Exception('Invalid tensor shape')

        return x

    def _enhance_PIL(self, x: Image.Image):
        x = ImageEnhance.Color(x)
        x = x.enhance(self.factor)
        return x

    def _enhance_Tensor(self, x: torch.Tensor):
        x = ToPILImage()(x)
        x = self._enhance_PIL(x)
        x = ToTensor()(x)
        return x

def color_enhacne_blend(x, factor = 1.5):
    x_g = Grayscale(3)(x)
    out = x_g * (1.0 - factor) + x * factor
    out[out < 0] = 0
    out[out > 1] = 1
    return out
