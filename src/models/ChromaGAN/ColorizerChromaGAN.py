import math
from models import Colorizer
import sys
import os
from flask import jsonify
import cv2
from keras import applications
from keras.models import load_model
import numpy as np
import tensorflow as tf

# data information
IMAGE_SIZE = 224
NUM_EPOCHS = 5


class ColorizerChromaGAN(Colorizer.Colorizer):
    UPLOAD_FOLDER = ""

    def __init__(self, inputFolderPath, colorizationName, hyperParameters) -> None:
        """
        Class that handles image colorization by using the ChromaGAN implementation from this repository : https://github.com/pvitoria/ChromaGAN
        Colorizing using this method works in batches : the images given as input are computed by batch of size given in hyper parameters

        inputFolderPath : string
            Path to folder of images to colorize
        colorizationName: string
            Name of the colorization
        hyperParameters : None
            "batch_size": int : Number of images to colorize in each batch (this is NOT the total images count)
        """
        super().__init__(inputFolderPath, colorizationName, hyperParameters)
        self.model = "ChromaGAN.h5"

        self.filelist = os.listdir(self.inputFolderPath)
        self.filelist = [f for f in self.filelist if not f.startswith('.')]

        self.batch_size = hyperParameters["batch_size"]
        self.filesSize = len(self.filelist)
        self.data_index = 0
        self.model_path = os.path.join("./models_ai/"+self.model)

    def getColorizerName(self):
        return "ChromaGAN"

    def innerColorize(self):
        # Load .h5 model file
        colorizationModel = load_model(self.model_path)

        # Loop over the batches count : each batch is a subset of input images that will be colorized by the same call to colorization model
        batches_count = math.ceil(self.filesSize/self.batch_size)
        for b in range(batches_count):
            # Generate the next batch to be colorized from the input images
            try:
                batchX, batchY,  filelist, original, labimg_oritList = self.generate_next_batch()
            except Exception as e:
               raise Exception("Failed to generate batch: {}\n".format(e))
                

            # Start the current batch colorization using the chromagan model
            predY, _ = colorizationModel.predict(np.tile(batchX, [1, 1, 1, 3]))

            # For each resulting image of the colorized batch
            for i in range(len(batchX)):

                # Process some transformations in order to be saved on disk
                originalResult = original[i]
                height, width, channels = originalResult.shape
                predictedAB = cv2.resize(
                    self.deprocess(predY[i]), (width, height))
                labimg_ori = np.expand_dims(labimg_oritList[i], axis=2)
                predResult = self.reconstruct_no(
                    self.deprocess(labimg_ori), predictedAB)

                # Save the resulting image to the given path
                save_path = os.path.join(
                    self.outputFolderPath, filelist[i][:-4] + "_colorized.jpg")
                cv2.imwrite(save_path, predResult)

    # Some utility methods taken and modified from the chromagan implementation github usage example

    def generate_next_batch(self):
        """
        Uses the images given as input to the colorizer to generate the next batch to colorize
        """
        batch = []
        labels = []
        filelist = []
        labimg_oritList = []
        originalList = []
        for _ in range(self.batch_size):

            # If self.data_index reaches the number of images to colorize, we end the batch generation
            if self.data_index >= self.filesSize:
                break

            imagePath = os.path.join(
                self.inputFolderPath, self.filelist[self.data_index])
            filelist.append(self.filelist[self.data_index])
            greyimg, colorimg, original, labimg_ori = self.read_img(imagePath)
            batch.append(greyimg)
            labels.append(colorimg)
            originalList.append(original)
            labimg_oritList.append(labimg_ori)
            self.data_index = self.data_index + 1 

        batch = np.asarray(batch)/255
        labels = np.asarray(labels)/255
        originalList = np.asarray(originalList , dtype=object)
        labimg_oritList = np.asarray(labimg_oritList , dtype=object)/255
        return batch, labels, filelist, originalList, labimg_oritList

    def deprocess(self, imgs):
        """
        This method is used to transform a resulting image in order to save it on disk
        This is a part of chromagan repository
        """
        imgs = imgs * 255
        imgs[imgs > 255] = 255
        imgs[imgs < 0] = 0 
        return imgs.astype(np.uint8)

    def reconstruct_no(self, batchX, predictedY):
        """
        This method is also used to transform a resulting image in order to save it on disk
        This is a part of chromagan repository
        """
        result = np.concatenate((batchX, predictedY), axis=2)
        result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
        return result

    def read_img(self, imagePath):
        """
        Read the image file and change the image color space as needed
        This is a part of chromagan repository
        """
        img = cv2.imread(imagePath, 3)
        labimg = cv2.cvtColor(cv2.resize(
            img, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2Lab)
        labimg_ori = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        return np.reshape(labimg[:, :, 0], (IMAGE_SIZE, IMAGE_SIZE, 1)), labimg[:, :, 1:], img, labimg_ori[:, :, 0]
