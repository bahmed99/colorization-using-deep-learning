import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.SuperAttention.data import *
from models.SuperAttention import super_unet_v1 , super_unet_v2
from models.SuperAttention.super_unet_v1 import *
from models.SuperAttention.super_unet_v2 import *
from models import ColorizerReference
device = "cpu"

class ColorizerSuperUnet(ColorizerReference.ColorizerReference):
    def __init__(self,inputFolderPath,colorizationName,hyperParameters):
        # Appel du constructeur du classe parent
        super().__init__( inputFolderPath, colorizationName, hyperParameters)
        self.inputFolderPath=inputFolderPath
        self.filelist = os.listdir(self.inputFolderPath)
        self.filelist = [f for f in self.filelist if not f.startswith('.')]
        self.filesSize = len(self.filelist)
        self.referencePath=self.hyperParameters["referencePath"]

        self.referenceList=os.listdir(self.referencePath)

        if(self.filesSize!=len(self.referenceList)):
            super().handleException()
            raise Exception("The number of images and references must be the same")

    def setupUnetModel(self):
        pass

    def innerColorize(self):
        # Loading images
      
        colorspace = 'lab'
        dataset = ReferenceDataset(self.referencePath,self.inputFolderPath, slic_target=None, transform=None,
                                target_transfom=ToTensor(), color_space=colorspace)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)

        # Load model
        if self.SUPER_VERSION == 1:
            MODEL_PATH = 'models_ai/SuperUnetV1.pt'
            model = super_unet_v1.gen_color_stride_vgg16_new(2)
        else : 
            MODEL_PATH = 'models_ai/SuperUnetV2.pt'
            model = super_unet_v2.gen_color_stride_vgg16_new(2)

        model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['state_dict'])
        model.to(device)
        model.eval()

        for param_color in model.parameters():
            param_color.requires_grad = False

        with torch.no_grad():
            for idx, (img_rgb_target, img_target_gray, ref_rgb, ref_gray, target_slic, ref_slic_all, img_ref_ab, img_gray_map, gray_real, ref_real) in enumerate(dataloader):

                # Target data
                img_gray_map = (img_gray_map).to(device=device, dtype=torch.float)
                img_target_gray = (img_target_gray).to(device=device, dtype=torch.float)
                gray_real = gray_real.to(device=device, dtype=torch.float)
                target_slic = target_slic

                # Loading references
                ref_rgb_torch = ref_rgb.to(device=device, dtype=torch.float)
                img_ref_gray = (ref_gray).to(device=device, dtype=torch.float)

                # VGG19 normalization
                img_ref_rgb_norm = imagenet_norm(ref_rgb_torch, device)
                img_target_gray_norm = img_target_gray
                img_ref_gray_norm = img_ref_gray

                ab_pred, pred_Lab_torch, pred_RGB_torch = \
                    model(img_target_gray_norm, img_ref_gray_norm, img_target_gray,
                        target_slic, ref_slic_all, img_gray_map, gray_real, img_ref_rgb_norm,
                        device)

                image_path = self.outputFolderPath+'/{:02d}_colorisee.png'.format(idx)
                
                save_image(pred_RGB_torch, image_path, normalize=True)


        