import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error

def SSIM(image1, image2):
    """computes and returns the SSIM value between image1 and image2"""
    
    assert image2.shape == image1.shape, "SSIM : images should have the same size" 
    score = structural_similarity(image1, image2, data_range=255, channel_axis = 2)
    
    return score

def PSNR(image1, image2):
    """computes and returns the PSNR value between image1 and image2"""
    
    assert image2.shape == image1.shape, "PSNR : images should have the same size" 

    score = peak_signal_noise_ratio(image1, image2)
    
    return score

def meanSquaredError(image1, image2):
    """computes and returns the MSE value between image1 and image2"""
    
    assert image2.shape == image1.shape, "MSE : images should have the same size" 
    
    score = mean_squared_error(image1, image2)
    
    return score

def meanAbsoluteError(image1, image2):
    
    assert image2.shape == image1.shape, "MAE : images should have the same size" 

    score = np.sum(np.absolute((image1.astype("float") - image2.astype("float"))))/(image1.shape[0]*image1.shape[1])
    
    return score
