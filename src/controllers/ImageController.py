from flask import make_response, request
from flask import Blueprint, jsonify
import time
from models import ColorizersHandler, ImageModel

import os
import cv2
import json
import base64

image_api = Blueprint('image_api', __name__)


def launch_colorisation(models, inputPath, colorizationName, referencePath="", scribblePath="", metrics=None):
    """
        models : list of models to execute
        inputPath : path to the folder of images to colorize
        colorizationName : name of the folder where the colorized images will be saved
        referencePath : path to the folder of reference images
        scribblePath : path to the folder of scribble images
        nb_images : number of images to colorize

        Launches the colorisation of the images of the dataset with every method in models
    """
    try:
        finalMetrics = {}
        for model in models:  # for every model
            # TODO: Set this parameter from interface
            hyperParameters = {"batch_size": 3, 'referencePath': referencePath,
                               'metrics': metrics, "scribblePath": scribblePath}

            # Create an instance of the corresponding class with right parameters
            model_colorize = ColorizersHandler.getColorizerClassFromName(model)(
                inputPath,
                colorizationName,
                hyperParameters
            )
            model_colorize.colorize()  # Launches colorizations
            if (metrics != []):
                metricsComputed = model_colorize.computeMetrics()  # compute metrics for the method
                if (metricsComputed == {}):
                    model_colorize.handleException()
                    raise Exception(
                        "Metrics could not be computed because groundtruth images are missing")

                # append metrics to the final dictionnary
                finalMetrics[model_colorize.getColorizerName()
                             ] = metricsComputed

            # save metrics to json file
            saveMetrics(finalMetrics, colorizationName)

        return "success"

    except Exception as e:
        print("Colorization Error Detected")
        print(e)
        return str(e)


def saveMetrics(metrics, colorizationName):
    """saves the metrics to the json description file of the colorization colorizationName"""

    with open(os.path.abspath(ImageModel.UPLOAD_FOLDER+"/"+ImageModel.RESULTS_FOLDER+colorizationName+"/description.json"), "r") as descriptionFile:
        fileData = json.load(descriptionFile)
        fileData["metrics"] = metrics
        descriptionFile.close()
    with open(os.path.abspath(ImageModel.UPLOAD_FOLDER+"/"+ImageModel.RESULTS_FOLDER+colorizationName+"/description.json"), "w") as descriptionFile:
        json.dump(fileData, descriptionFile, indent=4)
        descriptionFile.close()


@image_api.route("/colorize", methods=["POST"])
def colorize():
    files = request.files.getlist("file[]") or []
    dirs = request.files.getlist("dir[]") or []
    references = request.files.getlist("reference[]") or []
    scribble = request.form.get("scribble")
    scribblePath = os.path.abspath("view/public/uploads/Scribbles/"+scribble)
    # get the models specified by user
    models = request.form.getlist("model[]")
    colorization_name = request.form.get("colorization_name")
    metrics=request.form.getlist("metrics[]") or []
    files += dirs
    # If one of the input is empty a file with no name is posted, so filter it
    files = list(filter(lambda f: f.filename != "", files))

    json_file, name_folder, nb_images = ImageModel.save_images(
        colorization_name, files, models, "",reference=references, scribble=scribble)
    with open(os.path.abspath(ImageModel.UPLOAD_FOLDER+"/"+ImageModel.RESULTS_FOLDER+colorization_name+"/description.json"), "w") as outfile:
        json.dump(json_file, outfile)


    referencePath = "view/public/uploads/"+ImageModel.RESULTS_FOLDER+colorization_name+"/References"

    msg = launch_colorisation(
        models, name_folder, colorization_name, referencePath, scribblePath,metrics)

    if msg != "success":
        return jsonify({"message": "failed", "text": msg}), 500

    return jsonify({"message": "success", "name": colorization_name}), 200


@image_api.route("/loadImagesOriginForScribbles", methods=["POST"])
def loadImagesOriginForScribbles():
    files = request.files.getlist("file[]") or []
    dirs = request.files.getlist("dir[]") or []
    colorization_name = request.form.get("colorization_name")
    files += dirs
    path = request.form.get("path")
    return genericLoadImagesOrigin(files, colorization_name, path, ImageModel.UPLOAD_FOLDER+"/Scribbles/"+colorization_name+"/description.json", scribble=colorization_name)


@image_api.route("/loadImagesOrigin", methods=["POST"])
def loadImagesOrigin():
    files = request.files.getlist("file[]") or []
    dirs = request.files.getlist("dir[]") or []
    colorization_name = request.form.get("colorization_name")
    files += dirs
    path = request.form.get("path")
    return genericLoadImagesOrigin(files,colorization_name,path,ImageModel.UPLOAD_FOLDER+"/"+ImageModel.RESULTS_FOLDER+colorization_name+"/description.json")

def genericLoadImagesOrigin(files, colorization_name, path, json_path, scribble=""):
    '''
    Loads original images without colorization processing and create a Json file
    '''
    # If one of the input is empty a file with no name is posted, so filter it
    files = list(filter(lambda f: f.filename != "", files))
    json_file, a, b = ImageModel.save_images(
        colorization_name, files, path=path, scribble="", useScribbleForGenerateJson=True)

    with open(os.path.abspath(json_path), "w") as outfile:
        json.dump(json_file, outfile)

    return jsonify({"message": "success", "name": colorization_name}), 200


@image_api.route('/useKnownDatasetForScribbles', methods=['POST'])
def use_dataset_for_scribbles():
    dataset = request.json.get("dataset")
    colorization_name = request.json.get("colorization_name")
    return generic_use_dataset(dataset, colorization_name, ImageModel.UPLOAD_FOLDER+"/Scribbles/"+colorization_name+"/description.json")


@image_api.route('/useKnownDataset', methods=['POST'])
def use_dataset():
    dataset = request.json.get("dataset")
    colorization_name = request.json.get("colorization_name")
    return generic_use_dataset(dataset,colorization_name,ImageModel.UPLOAD_FOLDER+"/"+ImageModel.RESULTS_FOLDER+colorization_name+"/description.json")

def generic_use_dataset(dataset, colorization_name, json_path):
    '''
    Find datasets already created
    '''

    files = os.listdir(os.path.abspath("view/public/uploads/Datasets/"+dataset))

    files = [f for f in files if not f.startswith('.')]

    json_file = ImageModel.generate_json(
        colorization_name,  files, models=[], dataset=dataset, reference="", scribble="", useScribbleForGenerateJson=True)

    with open(os.path.abspath(json_path), "w") as outfile:
        json.dump(json_file, outfile)
    return jsonify({"message": "success", "name": colorization_name}), 200

# Get already done colorizations from view/upload folder
# used in the client side to display the list of colorisation


@image_api.route('/getColorizations', methods=['GET'])
def list_colorisations():

    upload_directory = './view/public/uploads'
    folders = os.listdir(os.path.abspath(upload_directory))
    folders = [d for d in folders if d != "Datasets" and d !=
               "Image_origin" and d != "References" and d != "Scribbles" and not (d.startswith('.'))]
    
    directory = './view/public/uploads/Results'
    folders = os.listdir(os.path.abspath(directory))
    data = []
    for f in folders:
        json_path = os.path.abspath('view/public/uploads/'+ImageModel.RESULTS_FOLDER+f)+"/description.json"
        json_object = []
        if os.path.exists(json_path):
            with open(json_path, 'r') as openfile:
                json_object = json.load(openfile)
        else:
            json_object = {
                "date": "_",
                "length_images": "Unknown",
                "models": ["Not  specified"],
                "use_scribbles": False,
            }
        data.append({
            "name": f,
            "date": json_object["date"],
            "length_images": json_object["length_images"],
            "models": json_object["models"],

        })

    return jsonify(data)

# Get scribbles list from view/upload folder
# used in the client side to display the list of scribbles


@image_api.route('/getScribbles', methods=['GET'])
def list_scribbles():

    directory = './view/public/uploads/Scribbles'
    folders = os.listdir(os.path.abspath(directory))
    data = []
    for f in folders:
        json_path = os.path.abspath(
            'view/public/uploads/Scribbles/'+f)+"/description.json"
        json_object = []
        if os.path.exists(json_path):
            with open(json_path, 'r') as openfile:
                json_object = json.load(openfile)
        else:
            json_object = {
                "date": "_",
                "length_images": "Unknown",
                "models": ["Not  specified"],
            }

        if "use_scribbles" in json_object and json_object['use_scribbles'] == True:
            data.append({
                "name": f,
                "date": json_object["date"],
                "length_images": json_object["length_images"],
                "models": json_object["models"],

            })
    return jsonify(data)


@image_api.route('/dataset', methods=['GET'])
def list_dataset():

    directory = os.path.abspath('./view/public/uploads/Datasets')

    # Create Dataset directory if it does not already exists
    os.makedirs(directory, exist_ok=True)

    files = os.listdir(directory)
    # Filter out directories whose names end with '_groundtruth' to avoid displaying groundtruth directories
    files = [f for f in files if not f.startswith(
        '.') and not f.endswith('_groundtruth')]
    return jsonify(files)


@image_api.route('/colorizeDataset', methods=['POST'])
def colorize_dataset():

    dataset = request.json.get("dataset")
    colorization_name = request.json.get("colorization_name")
    models = request.json.get("models")
    datasetPath = os.path.abspath("view/public/uploads/Datasets/"+dataset)
    reference= request.json.get("reference")
    scribble= request.json.get("scribble")

    referencePath = "view/public/uploads/References/"+reference
    scribblePath = os.path.abspath("view/public/uploads/Scribbles/"+scribble)

    files = os.listdir(datasetPath)
    files.sort()

    files = [f for f in files if not f.startswith('.')]
    json_file = ImageModel.generate_json(

        colorization_name, files, models, dataset,reference,scribble)
    

    with open(os.path.abspath(ImageModel.UPLOAD_FOLDER+"/"+ImageModel.RESULTS_FOLDER+colorization_name+"/description.json"), "w") as outfile:
        json.dump(json_file, outfile)

    # Get the selected metrics
    metrics = request.json.get("metrics[]")
    msg = launch_colorisation(
        models, datasetPath, colorization_name, referencePath, scribblePath, metrics)

    if msg != "success":

        return jsonify({"message": "failed", "text": msg}), 200

    return jsonify({"message": "success", "name": colorization_name}), 200


@image_api.route('/getModels', methods=['GET'])
def list_models():
    return jsonify(ColorizersHandler.listModels())


@image_api.route('/getImages/<colorization_name>', methods=['GET'])
def list_images(colorization_name):
    directory_json = os.path.abspath('view/public/uploads/'+ImageModel.RESULTS_FOLDER+"/"+colorization_name)
    with open(directory_json+"/description.json", 'r') as openfile:
        json_object = json.load(openfile)
    json_object["images_colorized"] = []
    i = 0
    for model in json_object["models"]:
        images = os.listdir(os.path.join(directory_json, model))
        images.sort()
        json_object["images_colorized"].append({})
        json_object["images_colorized"][i][model] = []
        for image in images:
            json_object["images_colorized"][i][model].append(
                os.path.join("/uploads/Results", colorization_name, model, image))
        i = i+1

    # If the colorization was done using a dataset, we display grountruth images if they exist
    groundtruthFolder = (
        json_object["images_origin_prefix"][:-1]
        # remove the last / from the images_origin_prefix
        if json_object["images_origin_prefix"][-1] == "/"
        else json_object["images_origin_prefix"]) + '_groundtruth'
    if json_object["use_dataset"] and os.path.exists(groundtruthFolder):
        json_object["images_ground_truth"] = []
        # This line is necessary since the path that uses the view is not relative to same folder as the backend
        viewGTFolderPath = "/uploads/" + \
            groundtruthFolder.split("/view/public/uploads/")[1]
        for gtImage in os.listdir(groundtruthFolder):
            json_object["images_ground_truth"].append(
                os.path.join(viewGTFolderPath, gtImage))
    return json_object


@image_api.route('/getImagesOrigin/<colorization_name>', methods=['GET'])
def list_images_origin(colorization_name):
    '''
    Get the list of original images
    '''
    directory_json = 'view/public/uploads/'+ImageModel.RESULTS_FOLDER+colorization_name+'/description'
    with open(os.path.abspath(directory_json)+".json", 'r') as openfile:
        json_object = json.load(openfile)

    return json_object


@image_api.route('/getImagesOriginForScribbles/<colorization_name>', methods=['GET'])
def list_images_origin_for_scribbles(colorization_name):
    '''
    Get the list of original images
    '''
    directory_json = 'view/public/uploads/Scribbles/'+colorization_name+'/description'
    with open(os.path.abspath(directory_json)+".json", 'r') as openfile:
        json_object = json.load(openfile)

    return json_object


@image_api.route('/saveImage/<colorization_name>/<fileName>', methods=['POST'])
def save_image(colorization_name, fileName):
    '''
    Method that calls the static method save_images to save the scribbles created
    '''
    data = request.json
    data_url = data['dataURL']
    # remove data:image/png;base64 from the beginning of the dataURL
    image_data = base64.b64decode(data_url[22:])

    image_name = ImageModel.save_scribble(
        colorization_name, image_data, fileName)
    directory_json = 'view/public/uploads/Scribbles/'+colorization_name+'/description'
    with open(os.path.abspath(directory_json)+".json", 'r') as openfile:
        json_object = json.load(openfile)

    path = "/uploads/Scribbles/"+colorization_name+"/"+image_name
    if not any(el == path for el in json_object['scribbles']) :
        json_object['scribbles'].append(
            path)
        json_object['use_scribbles'] = True

    with open(os.path.abspath(directory_json)+".json", 'w') as openfile:
        json.dump(json_object, openfile)

    # return jsonify({'success': True}), 200
    return json_object['scribbles']


@image_api.route("/checkColorizationName", methods=["POST"])
def check():
    colorization_name = request.json.get("colorization_name")
    if os.path.exists(os.path.abspath(ImageModel.UPLOAD_FOLDER+"/"+ImageModel.RESULTS_FOLDER+colorization_name)):
        return jsonify({"message": True}), 200
    return jsonify({"message": False}), 200


@image_api.route("/getReferences", methods=["GET"])
def get_references():
    '''
        Get the list of references
    '''
    directory = os.path.abspath('./view/public/uploads/References')

    files = os.listdir(directory)

    files = [f for f in files if not f.startswith('.')]

    return jsonify({"data": files})


@image_api.route("/getScribblesList", methods=["GET"])
def get_scribbles():
    '''
        Get the list of scribbles
    '''
    directory = os.path.abspath('./view/public/uploads/Scribbles')
    # list files if path exists
    if not os.path.exists(directory):
        return jsonify({"data": []})

    files = os.listdir(directory)

    files = [f for f in files if not f.startswith('.')]

    return jsonify({"data": files})
