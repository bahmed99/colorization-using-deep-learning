
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import skimage
from skimage import color
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from models.Colorizer import Colorizer
import torch.utils.model_zoo as model_zoo
from models.ZhangModels.ZhangMethodsUtils import preprocess_img, postprocess_tens, norm_layer, load_img, unnormalize_ab, normalize_l

'''
this class contains the architecture of the model that created with adding multiples layers 
(the implementation is from https://github.com/richzhang/colorization)
'''
class ColorizerECCV16(Colorizer, nn.Module):
    def __init__(self,inputFolderPath,colorizationName,hyperParameters ):
        # Appel des constructeurs des classes parentes
        Colorizer.__init__(self, inputFolderPath, colorizationName, hyperParameters)
        nn.Module.__init__(self)
        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.
        self.inputFolderPath=inputFolderPath
        self.filelist = os.listdir(self.inputFolderPath)
        self.filelist = [f for f in self.filelist if not f.startswith('.')]
        self.filesSize = len(self.filelist)


        model1=[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]

        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]

        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]

        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        model8=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]

        model8+=[nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

        model = "ECCV16.pth"
        model_path = os.path.join("./models_ai/"+ model)
        #loads the parameters (weights) of a PyTorch model from ECCV16.pth file
        weights = torch.load(model_path, map_location='cpu') #loads the parameters (weights) of a PyTorch model from ECCV16.pth file
        #self.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',map_location='cpu',check_hash=True))
        self.load_state_dict(weights)

    def forward(self, input_l):
        ''' 
        This method defines the forward pass of the model. 
        It takes an input tensor and passes it through the convolutional layers to produce an output tensor. 
        The output tensor represents the colorized version of the input image.
        '''
        conv1_2 = self.model1(normalize_l(input_l, self.l_cent, self.l_norm))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return unnormalize_ab(self.upsample4(out_reg), self.ab_norm)

    def getColorizerName(self):
        return "ECCV16"

    def innerColorize(self):
        ''' 
        added this function to adapt the call to our architecture ... it gets the model from the class 
        and foreach image it applies the model on this selected image and for now it save it in the current directory
        '''
        colorizer_eccv16 = self.eval()
        for i in range(self.filesSize) : 
            img = load_img(self.inputFolderPath + "/" +self.filelist[i])
            (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
            out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
            save_path = os.path.join(
                self.outputFolderPath, self.filelist[i][:-4] + "_colorized.png")
            plt.imsave(save_path, out_img_eccv16)






    