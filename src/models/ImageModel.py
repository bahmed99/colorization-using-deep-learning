from werkzeug.utils import secure_filename
import cv2
import os
import time
from datetime import datetime
import json
import os
from PIL import Image


class ImageModel():
    # TODO : find a way to store config variables globally
    UPLOAD_FOLDER = ""
    DATASETS_FOLDER = ""
    RESULTS_FOLDER = ""

    @staticmethod
    def save_images(colorization_name, files, models=[], path="", reference=[], scribble="",useScribbleForGenerateJson=False):

        if useScribbleForGenerateJson :
            os.mkdir(os.path.abspath(
                ImageModel.UPLOAD_FOLDER+"/Scribbles/"+colorization_name))
        else :
            os.mkdir(os.path.abspath(
                ImageModel.UPLOAD_FOLDER+"/"+ImageModel.RESULTS_FOLDER+colorization_name))

        save = ""
        images = []

        images_reference = []
        images_scribble = []

        for model in models:
            os.mkdir(os.path.abspath(
                ImageModel.UPLOAD_FOLDER+"/"+ImageModel.RESULTS_FOLDER+colorization_name+"/"+model))
        dataset_folder_name = "dataset{}".format(
            len(os.listdir(os.path.abspath(ImageModel.UPLOAD_FOLDER+"Datasets")))+1)

        if (len(reference) > 0):
                os.mkdir(os.path.abspath(
                    ImageModel.UPLOAD_FOLDER+"/"+ImageModel.RESULTS_FOLDER+colorization_name+"/References"))
                
        if(scribble != "") :
            scribble_list = [ImageModel.UPLOAD_FOLDER+"Scribbles/"+scribble+'/' +
                            i for i in os.listdir(os.path.abspath(ImageModel.UPLOAD_FOLDER+"Scribbles/"+scribble)) if i!="description.json"]
            scribble_list= [i.split("view/public")[1] for i in scribble_list]
            scribble_list.sort()

        if (path != ""):
            save, images = ImageModel.create_symbolic_link(
                path, dataset_folder_name)
        else:
            save, images = ImageModel.save_image_from_uploads(
                files, dataset_folder_name, colorization_name)
            if (len(reference) > 0):
                images_reference = ImageModel.save_image_from_reference(
                    reference, colorization_name)

        json_file = {
            "colorization_name": colorization_name,
            "models": models,
            "images_origin_prefix": save,
            "images_origin": images,
            "use_dataset": path != "" or len(files) > 20,
            "use_symbolic_link": path != "",
            "symbolic_link": path,
            "length_images": len(files) if path == "" else len(os.listdir(os.path.abspath(ImageModel.UPLOAD_FOLDER+"Datasets/"+dataset_folder_name))),
            "use_scribbles": scribble!="",
            "scribbles": images_scribble,
            "use_reference": len(reference) > 0,
            "reference": images_reference,
            "date": datetime.now().strftime("%Y_%m_%d %H:%M:%S")
        }

        return json_file, save, len(files) if path == "" else len(os.listdir(os.path.abspath(ImageModel.UPLOAD_FOLDER+"Datasets/"+dataset_folder_name)))

    @staticmethod
    def save_scribble(colorization_name, file, fileName):
        if not os.path.exists(os.path.abspath(ImageModel.UPLOAD_FOLDER+"/Scribbles/"+colorization_name)):
            os.mkdir(os.path.abspath(
                ImageModel.UPLOAD_FOLDER+"/Scribbles/"+colorization_name))

        save = os.path.abspath(os.path.join(
            ImageModel.UPLOAD_FOLDER+ "/Scribbles/"+colorization_name))
        image_name = "{}.{}".format(
            fileName.split(".")[0], "png")

        with open(os.path.join(save, image_name), "wb") as f:
            f.write(file)

        return image_name

    @staticmethod
    def generate_json(colorization_name, files, models, dataset, reference, scribble="",useScribbleForGenerateJson=False):
        save = ImageModel.UPLOAD_FOLDER + "Datasets/" + dataset

        images = [(save+'/'+f).split("view/public")[1] for f in files]
        reference_list = []
        scribble_list = []
        if useScribbleForGenerateJson:
            os.mkdir(os.path.abspath(
                ImageModel.UPLOAD_FOLDER+"/Scribbles/"+colorization_name))
        else:
            os.mkdir(os.path.abspath(
                ImageModel.UPLOAD_FOLDER+"/"+ImageModel.RESULTS_FOLDER+colorization_name))
            for model in models:
                os.mkdir(os.path.abspath(
                    ImageModel.UPLOAD_FOLDER+"/"+ImageModel.RESULTS_FOLDER+colorization_name+"/"+model))
        if(reference != "") :     
            reference_list = [ImageModel.UPLOAD_FOLDER+"References/"+reference+'/' +
                            i for i in os.listdir(os.path.abspath(ImageModel.UPLOAD_FOLDER+"References/"+reference))]
            reference_list= [i.split("view/public")[1] for i in reference_list]
            reference_list.sort()
        
        if(scribble != "") :
            scribble_list = [ImageModel.UPLOAD_FOLDER+"Scribbles/"+scribble+'/' +
                            i for i in os.listdir(os.path.abspath(ImageModel.UPLOAD_FOLDER+"Scribbles/"+scribble)) if i!="description.json"]
            scribble_list= [i.split("view/public")[1] for i in scribble_list]
            scribble_list.sort()


        json_file = {
            "colorization_name": colorization_name,
            "models": models,
            "images_origin_prefix": save,
            "images_origin": images,
            "images_colorized": [],
            "use_dataset": True,
            "use_symbolic_link": True,
            "symbolic_link": '',
            "length_images":  len(os.listdir(os.path.abspath(ImageModel.UPLOAD_FOLDER+"Datasets/"+dataset))),
            "use_scribbles": scribble!="",
            "scribbles": scribble_list,
            "use_reference": reference != "",
            "reference":  reference_list if reference != "" else [],
            "date": datetime.now().strftime("%Y_%m_%d %H:%M:%S")
        }
        return json_file

    @staticmethod
    def save_files_json(name, files, dataset_folder_name):

        # file_name = name+"_{}_{}_{}_{}:{}:{}".format(time.localtime().tm_mday, time.localtime(
        # ).tm_mon, time.localtime().tm_year, time.localtime().tm_hour, time.localtime().tm_min, time.localtime().tm_sec)
        file_name = name+"_"+datetime.now().strftime("%Y_%m_%d %H:%M:%S")
        json_file = {
            "files": []
        }
        for f in files:
            json_file["files"].append(dataset_folder_name+"/"+f)

        with open(os.path.abspath(ImageModel.UPLOAD_FOLDER+"Image_origin/"+f"{file_name}.json"), "w") as outfile:
            json.dump(json_file, outfile)

        return file_name

    @staticmethod
    def create_symbolic_link(path, dataset_folder_name):
        i = 0
        images = []
        save = ImageModel.UPLOAD_FOLDER + "Datasets/" + dataset_folder_name
        if not os.path.exists(os.path.abspath(save)):
            os.mkdir(os.path.abspath(save))

        files_list = os.listdir(os.path.abspath(path))

        for file in files_list:
            if (file.endswith(".jpg") or file.endswith(".png")):
                os.symlink(os.path.abspath(path+"/"+file), os.path.abspath(
                    save+"/"+f"image_{i}"))

                images.append((save+"/"+f"image_{i}").split("view/public")[1])
                i = i+1
        return save, images

    @staticmethod
    def save_image_from_uploads(files, dataset_folder_name, colorization_name):
        i = 0
        images = []
        save = ""
        for f in files:
                filename = secure_filename(f.filename)
                
                save = ImageModel.UPLOAD_FOLDER + "Image_origin/" + colorization_name
                if not os.path.exists(os.path.abspath(save)):
                    os.mkdir(os.path.abspath(save))

                filename = secure_filename(f.filename)
                if not filename.endswith('.DS_Store'):  #filtering out ".DS_Store" files and not storing them 
                    image_name = "image_{}.{}".format(i, filename.split(".")[-1])
                    images.append(os.path.join(
                        save, image_name).split("view/public")[1])
                    f.save(os.path.join(save, image_name))
                    # json_file["files"].append(dataset_folder_name+"/"+image_name)
                    i = i+1

        return save, images

    @staticmethod
    def save_image_from_reference(files, colorization_name):
        i = 0
        images = []
        save = ""
        for f in files:
            filename = secure_filename(f.filename)

            save = ImageModel.UPLOAD_FOLDER +"/"+ImageModel.RESULTS_FOLDER + colorization_name + "/References"
            if not os.path.exists(os.path.abspath(save)):
                os.mkdir(os.path.abspath(save))

            filename = secure_filename(f.filename)
            if not filename.endswith('.DS_Store'):  #filtering out ".DS_Store" files and not storing them 
                image_name = "image_{}.{}".format(i, filename.split(".")[-1])
                images.append(os.path.join(
                    save, image_name).split("view/public")[1])
                f.save(os.path.join(save, image_name))
                i = i+1

        return images
