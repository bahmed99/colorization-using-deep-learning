import os
import models.metrics as metrics
import cv2
import math
import shutil

class Colorizer:
    def __init__(self, inputFolderPath, colorizationName, hyperParameters) -> None:
        """
        inputFolderPath : string
            Path to folder of images to colorize
        colorizationName: string
            Name of the colorization
        hyperParameters : array
            Parameters that depends on the specific colorization method
        """
        self.inputFolderPath = inputFolderPath
        self.hyperParameters = hyperParameters
        
        self.colorizationName = colorizationName

        self.outputFolderPath = os.path.join(os.path.abspath("view/public/uploads/Results/"),
                                             colorizationName, self.getColorizerName())
        # Create output directory if it does not already exists
        os.makedirs(self.outputFolderPath, exist_ok=True)

    def colorize(self):
        """
        This method is the one to call in order to start a colorization.
        This is a wrapper over the innerColorize() method to properly execute the colorization depending on the model implementation.
        """

        try:
            self.innerColorize()
            print('\033[92m' + "======> Colorizer '" + self.getColorizerName() + "' task terminated." + '\033[0m')

        except Exception as e:
            
            print('\033[91m' + "======> Colorizer '" + self.getColorizerName() + "' task failed." + '\033[0m')
            self.handleException()
            raise e
       
    def innerColorize(self) -> None:
        """
        Method that executes the colorization depending on the method.
        Override this method with the proper code (adapted to a specified method).
        """
        pass

    def getColorizerName(self) -> None:
        pass
 

    def handleException(self):
        """
        Method to handle exceptions.

        """
        #delete colorization folder
        shutil.rmtree(os.path.join(os.path.abspath("view/public/uploads/Results"),
                                             self.colorizationName))
        
        
    
    def computeMetrics(self):
        """computes the metrics for all images colorized"""
        imagesMetricsDic = {} # where the metrics are stored
        images = os.listdir(self.outputFolderPath) # get the folder where colorized images are
        grayImages = os.listdir(self.inputFolderPath) 
        for f in grayImages:
            if f.startswith("."):
                grayImages.remove(f)  #removing files starting with '.' to resolve .DS_files bug  
        isDataset = (self.inputFolderPath).split("/")
 

        # to ensure images are treated in the right order
        images.sort()
        grayImages.sort()
        
        if (isDataset[len(isDataset)-2] == "Datasets" and os.path.isdir(self.inputFolderPath+"_groundtruth")): # if groundtruth exist
            for i in range(len(images)): # for every colorized image
                if os.path.isfile(self.inputFolderPath+"_groundtruth"+"/"+grayImages[i]): # if the groundtruth exists for the images
                    im1 = cv2.imread(self.outputFolderPath+"/"+images[i]) # load with opencv
                    im2 = cv2.imread(self.inputFolderPath+"_groundtruth"+"/"+grayImages[i]) # groundtruth
                    imagesMetricsDic[images[i]] = self.applyMetrics(im1, im2) # store metrics
                else :
                    imagesMetricsDic[images[i]] = None # no metrics because no groundtruth
                    
        return imagesMetricsDic
      

        

    def applyMetrics(self,img1,img2):
        """
        computes every metric selected between img1 and img2
        returns the results as a dictionnary
        
        img1: Image to evaluate
        img2: Ground truth image
        """

        metrics_dict={}
        for metric in self.hyperParameters['metrics']:
            if metric == "PSNR":
                score = metrics.PSNR(img1,img2) 
            elif metric == "MAE":
                score = metrics.meanAbsoluteError(img1,img2)
            elif metric == "MSE":
                score = metrics.meanSquaredError(img1,img2)
            elif metric == "SSIM":
                score = metrics.SSIM(img1,img2)
            else:
                continue
            if math.isinf(score):
                score = None
            metrics_dict[metric] = score 
        return metrics_dict
