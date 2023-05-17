from models.SuperAttention.data import *
from models.SuperAttention.super_unet_v2 import *
from models.SuperAttention import SuperUnet, super_unet_v2

class ColorizerSuperUnetV2(SuperUnet.ColorizerSuperUnet):

    def __init__(self,inputFolderPath,colorizationName,hyperParameters):
        super().__init__(inputFolderPath, colorizationName, hyperParameters)
        self.SUPER_VERSION = 2

    def getColorizerName(self):
        return "SuperUnetV2"