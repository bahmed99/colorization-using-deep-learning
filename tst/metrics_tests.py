import cv2
import sys 
import math
sys.path.append('../src/models/')
from metrics import SSIM, PSNR, meanSquaredError, meanAbsoluteError

class bcolors: # colors for display
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    GRAY = '\033[97m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def SSIM_test():
    
    print("")
    print(bcolors.GRAY + SSIM_test.__name__)
    print("")
    
    print(bcolors.OKCYAN + "SSIM on 2 differents images (png)")
    
    img1 = cv2.imread("img_test/bruschetta1.png")
    img2 = cv2.imread("img_test/bruschetta2.png")
    
    s = SSIM(img1, img2)
    
    if (s != 1):
        print(bcolors.OKGREEN + "test passed")
    else :
        print(bcolors.FAIL + "test failed")
    
    print(bcolors.OKCYAN + "SSIM on same images (png)")
    
    img1 = cv2.imread("img_test/bruschetta1.png")
    img2 = cv2.imread("img_test/bruschetta1.png")
    
    s = SSIM(img1, img2)
    
    if (s == 1):
        print(bcolors.OKGREEN + "test passed")
    else :
        print(bcolors.FAIL + "test failed")
    
    print(bcolors.OKCYAN + "SSIM on same jpeg images")
    
    img1 = cv2.imread("img_test/JPEG_example_flower.jpeg")
    img2 = cv2.imread("img_test/JPEG_example_flower.jpeg")
    
    s = SSIM(img1, img2)
    
    if (s == 1):
        print(bcolors.OKGREEN + "test passed")
    else :
        print(bcolors.FAIL + "test failed")
    
    print(bcolors.OKCYAN + "SSIM on same jpg images")
    
    img1 = cv2.imread("img_test/ocean.jpg")
    img2 = cv2.imread("img_test/ocean.jpg")
    
    s = SSIM(img1, img2)
    
    if (s == 1):
        print(bcolors.OKGREEN + "test passed")
    else :
        print(bcolors.FAIL + "test failed")
        
def PSNR_test():
    print("")
    print(bcolors.GRAY + PSNR_test.__name__)
    print("")
    
    print(bcolors.OKCYAN + "PSNR on 2 differents images (png)")
    
    img1 = cv2.imread("img_test/bruschetta1.png")
    img2 = cv2.imread("img_test/bruschetta2.png")
    
    s = PSNR(img1, img2)

    if (s != math.inf):
        print(bcolors.OKGREEN + "test passed")
    else :
        print(bcolors.FAIL + "test failed")
    
    print(bcolors.OKCYAN + "PSNR on same images (png)")
    
    img1 = cv2.imread("img_test/bruschetta1.png")
    img2 = cv2.imread("img_test/bruschetta1.png")
    
    s = PSNR(img1, img2)
    if (s == math.inf):
        print(bcolors.OKGREEN + "test passed")
    else :
        print(bcolors.FAIL + "test failed")
        
    print(bcolors.OKCYAN + "PSNR on same jpeg images")
    
    img1 = cv2.imread("img_test/JPEG_example_flower.jpeg")
    img2 = cv2.imread("img_test/JPEG_example_flower.jpeg")
    
    s = PSNR(img1, img2)
    
    if (s == math.inf):
        print(bcolors.OKGREEN + "test passed")
    else :
        print(bcolors.FAIL + "test failed")
        
    print(bcolors.OKCYAN + "PSNR on same jpg images")
    
    img1 = cv2.imread("img_test/ocean.jpg")
    img2 = cv2.imread("img_test/ocean.jpg")
    
    s = PSNR(img1, img2)
    if (s == math.inf):
        print(bcolors.OKGREEN + "test passed")
    else :
        print(bcolors.FAIL + "test failed")

def MSE_test():
    print("")
    print(bcolors.GRAY + MSE_test.__name__)
    print("")
    
    print(bcolors.OKCYAN + "MSE on 2 differents images (png)")
    
    img1 = cv2.imread("img_test/bruschetta1.png")
    img2 = cv2.imread("img_test/bruschetta2.png")
    
    mse = meanSquaredError(img1, img2)
    
    if (mse != 0):
        print(bcolors.OKGREEN + "test passed")
    else :
        print(bcolors.FAIL + "test failed")
        
    print(bcolors.OKCYAN + "MSE on same images (png)")
    
    img1 = cv2.imread("img_test/bruschetta1.png")
    img2 = cv2.imread("img_test/bruschetta1.png")
    
    mse = SSIM(img1, img2)

    if (mse == 1):
        print(bcolors.OKGREEN + "test passed")
    else :
        print(bcolors.FAIL + "test failed")
        
    print(bcolors.OKCYAN + "MSE on same jpeg images")
    
    img1 = cv2.imread("img_test/JPEG_example_flower.jpeg")
    img2 = cv2.imread("img_test/JPEG_example_flower.jpeg")
    
    mse = meanSquaredError(img1, img2)
    
    if (mse == 0):
        print(bcolors.OKGREEN + "test passed")
    else :
        print(bcolors.FAIL + "test failed")
    
    print(bcolors.OKCYAN + "MSE on same jpg images")
    
    img1 = cv2.imread("img_test/ocean.jpg")
    img2 = cv2.imread("img_test/ocean.jpg")
    
    mse = meanSquaredError(img1, img2)
    
    if (mse == 0):
        print(bcolors.OKGREEN + "test passed")
    else :
        print(bcolors.FAIL + "test failed")

def MAE_test():
    print("")
    print(bcolors.GRAY + MAE_test.__name__)
    print("")
    
    print(bcolors.OKCYAN + "MAE on 2 differents images (png)")
    
    img1 = cv2.imread("img_test/bruschetta1.png")
    img2 = cv2.imread("img_test/bruschetta2.png")
    
    mae = meanAbsoluteError(img1, img2)
    
    if (mae != 0):
        print(bcolors.OKGREEN + "test passed")
    else :
        print(bcolors.FAIL + "test failed")
        
    print(bcolors.OKCYAN + "MAE on same images (png)")
    
    img1 = cv2.imread("img_test/bruschetta1.png")
    img2 = cv2.imread("img_test/bruschetta1.png")
    
    mae = SSIM(img1, img2)

    if (mae == 1):
        print(bcolors.OKGREEN + "test passed")
    else :
        print(bcolors.FAIL + "test failed")
        
    print(bcolors.OKCYAN + "MAE on same jpeg images")
    
    img1 = cv2.imread("img_test/JPEG_example_flower.jpeg")
    img2 = cv2.imread("img_test/JPEG_example_flower.jpeg")
    
    mae = meanSquaredError(img1, img2)
    
    if (mae == 0):
        print(bcolors.OKGREEN + "test passed")
    else :
        print(bcolors.FAIL + "test failed")
    
    print(bcolors.OKCYAN + "MAE on same jpg images")
    
    img1 = cv2.imread("img_test/ocean.jpg")
    img2 = cv2.imread("img_test/ocean.jpg")
    
    mae = meanSquaredError(img1, img2)
    
    if (mae == 0):
        print(bcolors.OKGREEN + "test passed")
    else :
        print(bcolors.FAIL + "test failed")
        
    
if __name__ == "__main__":
    SSIM_test()
    PSNR_test()
    MSE_test()
    MAE_test()