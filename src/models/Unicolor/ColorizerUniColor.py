import os
from PIL import Image
import numpy as np

from models.Unicolor.sample.colorizer import Colorizer
from models.Unicolor.sample.utils_func import *
from models import ColorizerScribbles
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import skimage
from skimage import color
import matplotlib.pyplot as plt
import os


class ColorizerUniColor(ColorizerScribbles.ColorizerScribbles):

    def __init__(self, inputFolderPath, colorizationName, hyperParameters):
        super().__init__(inputFolderPath, colorizationName, hyperParameters)
        self.filelist = os.listdir(self.inputFolderPath)
        self.filelist = [f for f in self.filelist if not f.startswith('.')]
        self.filesSize = len(self.filelist)
        self.device = 'cpu'
        self.model = "mscoco_step259999"
        self.ckpt_file = os.path.join("./models_ai/"+self.model)
        self.scribblePath = self.hyperParameters["scribblePath"]
        self.scribbleList = os.listdir(self.scribblePath)
        self.scribbleList = [
            f for f in self.scribbleList if not f == "description.json"]
        
        # to make sure the right scribbles are used for the right image
        self.scribbleList.sort()
        self.filelist.sort()
        

        # Load CLIP and ImageWarper for text-based and exemplar-based colorization
        self.colorizer = Colorizer(self.ckpt_file, self.device, [
                                   256, 256], load_clip=False, load_warper=False)

    def innerColorize(self):
        
        for i in range(self.filesSize):
            img = Image.open(self.inputFolderPath + "/" + self.filelist[i]).convert('L')
            I_stk = self.colorizer.sample(
                img, self.getPointsFromScribbles(i), topk=100)
            save_path = os.path.join(
                self.outputFolderPath, self.filelist[i][:-4] + "_colorized.jpg")
            #save image with PIL
            I_stk.save(save_path)
            
        os.chdir('./src')

    def getPointsFromScribbles(self, i):
        points = []
        image = Image.open(self.scribblePath + "/" + self.scribbleList[i])
        largeur, hauteur = image.size

        for x in range(largeur):
            for y in range(hauteur):
                couleur = image.getpixel((x, y))
                if couleur[3] != 0: # if there is a colored pixel
                    points.append(
                        {'index': [(x*255)//largeur, (y*255)//hauteur], 'color': list(couleur[:3])})
        return points

    def getColorizerName(self):
        return "UniColor"
