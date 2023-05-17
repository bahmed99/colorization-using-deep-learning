from .ImageModel import *
from .ChromaGAN.ColorizerChromaGAN import *
import os

ImageModel.UPLOAD_FOLDER = os.path.abspath("view/public/uploads") +"/"
ColorizerChromaGAN.UPLOAD_FOLDER = os.path.relpath("view/public/uploads") +"/"
ImageModel.DATASETS_FOLDER = "Datasets/"
ImageModel.RESULTS_FOLDER = "Results/"
