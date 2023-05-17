import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity
import json


# -------------------- HELPER FUNCTIONS -------------------- #

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
    
def read_images_from_folder(folderName):
    
    """returns the list of images names in the folder as well as the list of images loaded with opencv"""
    
    images = os.listdir(folderName) # list file in folder
    imageNames = [image for image in images if not image.startswith('.')] # to deal with .DS_Store in MACOS computers
    
    imageNames.sort() # sort images by alphabetical names to make sure same images are compared
    
    imageList = [cv2.imread(folderName + i) for i in imageNames] # load images with openCV
    
    return imageNames, imageList

def getConfig(configFile):
    
    """reads and return the data from the file configFile"""
    
    file = open(configFile)
    data = json.load(file)
    file.close()
    
    return data

# -------------------- -------------------- -------------------- #

# -------------------- METRICS FUNCTIONS -------------------- #

def compare_images_pixel(image1, image2):
    
    """returns the number of values differents in the 2 images as well as the total number of values"""
    
    assert image2.shape == image1.shape, "images should have the same size"
    
    height = image1.shape[0]
    width = image1.shape[1]
    diff = cv2.subtract(image1, image2) 
    nb_different_values = np.count_nonzero(diff)
    nb_total_values = height*width
    
    return nb_different_values, nb_total_values*3

def proportion_of_different_values(image1, image2, proportion_accepted, image_name):
    
    """asserts the proportion of different values of the 2 images is inferior to proportion_accepted
    returns True if test passed ; False otherwise"""
    
    nb_different_values, nb_pixel_total = compare_images_pixel(image1, image2)
    
    proportion_different = nb_different_values/nb_pixel_total
    
    print(bcolors.OKBLUE + image_name, end = ' : ')
    
    if (proportion_different <= proportion_accepted):
        print(bcolors.OKGREEN + "OK test passed ")
        return True
    else :
        print(bcolors.FAIL + "NOK test failed ", round(proportion_different*100,2), "% values are different")
        return False
  
def SSIM(image1, image2, proportion_accepted, image_name):
    
    """computes the SSIM value between image1 and image2 and displays whether that value is under threshold or not
    returns True if test passed ; False otherwise"""
    
    assert image2.shape == image1.shape, "images should have the same size"
    
    score = structural_similarity(image1, image2, channel_axis = 2)
    
    print(bcolors.OKBLUE + image_name, end = ' : ')

    if (score >= proportion_accepted):
        print(bcolors.OKGREEN + " OK test passed ")
        return True
    else :
        print(bcolors.FAIL + "NOK test failed ", round(score*100,2), "% of similarity")
        return False
    
# -------------------- -------------------- -------------------- #

# -------------------- DISPLAY FUNCTIONS -------------------- #

def headerPrint(function):
    
    """displays the header for each test"""
    
    print("")
    print(bcolors.GRAY + function.__name__)
    print("")
    
def headerMethod(method):
    """displays the header for each method"""
    
    print("")
    print(bcolors.HEADER + "***************************************")
    print(bcolors.HEADER + "************* " + method + " *************")
    print(bcolors.HEADER + "***************************************")
    print("")
    
def summary(nb_total, nb_failed, nb_passed):
    """displays the summary of tests performed"""
    
    print("")
    print(bcolors.HEADER + "************* " + "SUMMARY" + " *************")
    print(bcolors.HEADER + "***************************************")
    print(bcolors.OKBLUE + str(nb_total) + " tests done")
    print(bcolors.OKGREEN + str(nb_passed) + " tests passed")
    print(bcolors.FAIL + str(nb_failed) + " tests failed")
    print("")

# -------------------- -------------------- -------------------- #
  
if __name__ == "__main__":
    
    config = getConfig("config.json") # get config from json file
    
    test_config = config["tests"]["thresholds"]
    tests = [] # list of test functions to execute
    thresholds = [] # list of the threshold accepted by each one of the test functions
    for test_func in test_config:
        tests.append(globals()[test_func]) # get function from its name
        thresholds.append(test_config[test_func])
        
    nb_failed = 0 # number of tests failed
    nb_passed = 0 # number of tests passed
    nb_total = 0 # number of tests performed
    
    methods = config["methods"]
    
    for method in methods:
        headerMethod(method)
    
        images_repository = read_images_from_folder(methods[method]["folder_repository"])
        images_interface = read_images_from_folder(methods[method]["folder_interface"])
        
        assert len(images_repository[0]) == len(images_interface[0]), "both folders should have the same number of images"
       
        test_index = 0
        for test in tests:
            headerPrint(test) # header display for the test
            for i in range(len(images_repository[0])): # for all images in the folder
                ret_val = test(images_repository[1][i], images_interface[1][i], thresholds[test_index], images_repository[0][i]) 
                
                # updates test statistics
                if (ret_val) : 
                    nb_passed +=1
                else :
                    nb_failed+=1
                nb_total+=1
                
            test_index += 1
            
    summary(nb_total, nb_failed, nb_passed)
        
        
    
    

    