import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from skimage import color
import torch.nn.functional as F

'''
Copyright (c) 2016, Richard Zhang, Phillip Isola, Alexei A. Efros
All rights reserved.

Github repository : https://github.com/richzhang/colorization
'''

def resize_img(img, HW=(256,256), resample=3):
    ''' 
    The "resize_img" function is a method for resizing an input image. 
    The function takes three parameters as inputs: 
    "img", which is the original image to be resized; 
    "HW", which is a tuple containing the desired height and width dimensions of the output image;
    "resample", which specifies the method used for resampling the image (defaulting to a value of 3, which corresponds to the Lanczos filter)
    '''
    return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
    '''
    This function preprocesses an RGB image by resizing it to a specified size using the resize_img function and converting it to the LAB color space. 
    It then extracts the L channel from the LAB image, converts it to PyTorch tensors, and returns the original size L and resized L as a tuple of tensors.
    The HW argument specifies the desired width and height of the resized image, and the resample argument controls the downsampling method used by the resize function.
    '''
    img_rgb_orig = img_rgb_orig[:, :, :3]  # Supprime le canal alpha
    # return original size L and resized L as torch Tensors
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_orig = img_lab_orig[:,:,0]
    img_l_rs = img_lab_rs[:,:,0]

    tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]
    tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]

    return (tens_orig_l, tens_rs_l)

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
    '''
    This function takes as input the original grayscale image and the predicted colorization in the AB color space, both as PyTorch tensors.
    It resizes the predicted colorization to match the size of the original grayscale image, concatenates it with the original grayscale image in the LAB color space,
    converts the resulting LAB image to RGB, and returns the resulting color image as a numpy array. 
    The mode argument specifies the interpolation mode to use when resizing the predicted colorization, and bilinear is the default.
    '''
    # tens_orig_l 	1 x 1 x H_orig x W_orig
    # out_ab 		1 x 2 x H x W

    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]

    # call resize function if needed
    if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
    else:
        out_ab_orig = out_ab

    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
    return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))

def load_img(img_path):
    '''
    This function loads an image from a given file path and returns a numpy array. 
    If the image is grayscale, it is converted to a 3-channel image by replicating the same channel three times.
    '''
    out_np = np.asarray(Image.open(img_path))
    if(out_np.ndim==2):
        out_np = np.tile(out_np[:,:,None],3)   
    return out_np

def norm_layer(num_channels):
    '''
    This function returns a batch normalization layer for 2D convolutional neural networks, given the number of channels in the input tensor.
    '''
    return nn.BatchNorm2d(num_channels)

def normalize_l(in_l, l_cent, l_norm):
    return (in_l-l_cent)/l_norm

def unnormalize_l(in_l, l_cent, l_norm):
    '''
    Not used by current zhang methods
    '''
    return in_l*l_norm + l_cent

def normalize_ab(in_ab, ab_norm):
    return in_ab/ab_norm

def unnormalize_ab(in_ab, ab_norm):
    return in_ab*ab_norm