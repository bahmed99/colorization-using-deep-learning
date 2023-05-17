from models.SuperAttention.data import *
from models.SuperAttention.super_unet_v1 import *
from models.SuperAttention import SuperUnet, super_unet_v1

class ColorizerSuperUnetV1(SuperUnet.ColorizerSuperUnet):

    def __init__(self,inputFolderPath,colorizationName,hyperParameters):
        super().__init__(inputFolderPath, colorizationName, hyperParameters)
        self.SUPER_VERSION = 1

    def getColorizerName(self):
        return "SuperUnetV1"