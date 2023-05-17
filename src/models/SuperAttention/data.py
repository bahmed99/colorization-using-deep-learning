from torch.utils.data import Dataset
from skimage.color import rgb2gray
from models.SuperAttention.utils import *
import glob


class ReferenceDataset(Dataset):
    def __init__(self, ref_image,target_image, slic_target, transform=None, target_transfom=None, slic=True, size=224, color_space=None):
        self.slic = slic
        self.target_path = sorted(glob.glob(target_image+'/*'))
        self.ref_path = sorted(glob.glob(ref_image+'/*'))

        self.len = len(self.target_path)
        self.transform = transform
        self.target_transform = target_transfom
        self.size = size
        self.color_space = color_space
        self.slic_target = slic_target

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x_real = rgb2gray( io.imread(self.target_path[index], pilmode='RGB'))
        x = rgb2gray(resize(io.imread(self.target_path[index], pilmode='RGB'), (224, 224)))

        ref_real = io.imread(self.ref_path[index], pilmode='RGB')
        ref = resize(io.imread(self.ref_path[index], pilmode='RGB'), (224, 224))

        if self.color_space == 'lab':
            x_luminance_classic = x
            ref_new_color = color.rgb2lab(ref)
            ref_luminance_classic = (ref_new_color[:, :, 0] / 100.0)
            ref_chroma = ref_new_color[:, :, 1:] / 127.0

            #Luminance remapping
            x_luminance_map = (np.std(ref_luminance_classic) / np.std(x_luminance_classic)) * (x_luminance_classic - np.mean(x_luminance_classic)) + np.mean(ref_luminance_classic)
            ref_luminance = ref_luminance_classic

        # Calculating superpixel label map for target and reference images (Grayscale)
        target_slic = img_segments_only(x_luminance_classic, 1, self.size)
        ref_slic = img_segments_only(ref_luminance, 1, self.size)

        target_slic_2 = img_segments_only(x_luminance_classic, 2, int(self.size / 2))
        ref_slic_2 = img_segments_only(ref_luminance, 2, int(self.size / 2))

        target_slic_3 = img_segments_only(x_luminance_classic, 4, int(self.size / 4))
        ref_slic_3 = img_segments_only(ref_luminance, 4, int(self.size / 4))

        target_slic_4 = img_segments_only(x_luminance_classic, 8, int(self.size / 8))
        ref_slic_4 = img_segments_only(ref_luminance, 8, int(self.size / 8))

        # Applying transformation (To tensor) and replicating tensor for gray scale images
        if self.target_transform:
            target_slic_all = []
            ref_slic_all = []

            x = self.target_transform(x)
            x_real = self.target_transform(x_real)
            ref_real = self.target_transform(ref_real)

            ref = self.target_transform(ref)

            target_slic_torch = self.target_transform(target_slic[:, :, np.newaxis])
            target_slic_torch_2 = self.target_transform(target_slic_2[:, :, np.newaxis])
            target_slic_torch_3 = self.target_transform(target_slic_3[:, :, np.newaxis])
            target_slic_torch_4 = self.target_transform(target_slic_4[:, :, np.newaxis])

            ref_slic_torch = self.target_transform(ref_slic[:, :, np.newaxis])
            ref_slic_torch_2 = self.target_transform(ref_slic_2[:, :, np.newaxis])
            ref_slic_torch_3 = self.target_transform(ref_slic_3[:, :, np.newaxis])
            ref_slic_torch_4 = self.target_transform(ref_slic_4[:, :, np.newaxis])

            target_slic_all.append(target_slic_torch)
            target_slic_all.append(target_slic_torch_2)
            target_slic_all.append(target_slic_torch_3)
            target_slic_all.append(target_slic_torch_4)

            ref_slic_all.append(ref_slic_torch)
            ref_slic_all.append(ref_slic_torch_2)
            ref_slic_all.append(ref_slic_torch_3)
            ref_slic_all.append(ref_slic_torch_4)

            x_luminance_map = self.target_transform(x_luminance_map[:, :, np.newaxis])
            x_luminance = self.target_transform(x_luminance_classic[:, :, np.newaxis])

            luminance_replicate_map = torch.cat((x_luminance_map.float(), x_luminance_map.float(), x_luminance_map.float()), dim=0)
            luminance_replicate = torch.cat((x_luminance.float(), x_luminance.float(), x_luminance.float()), dim=0)

            ref_luminance = self.target_transform(ref_luminance_classic[:, :, np.newaxis])
            ref_chroma = self.target_transform(ref_chroma)
            ref_luminance_replicate = torch.cat((ref_luminance.float(), ref_luminance.float(), ref_luminance.float()), dim=0)

        # Output: x: target image rgb, ref: reference image rdb, luminance_replicate: target grayscale image replicate, ref_luminance_replicate: reference grayscale image replicate
        # labels_torch: label map target image, labels_ref_torch: label map reference image, x_luminance: target grayscale image, ref_luminance: reference grayscale image.
        return x, luminance_replicate, ref, ref_luminance_replicate, target_slic_all, ref_slic_all, ref_chroma, luminance_replicate_map, x_real, ref_real


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img = sample
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2,0,1))
        img = torch.from_numpy(img)
        return img


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img

