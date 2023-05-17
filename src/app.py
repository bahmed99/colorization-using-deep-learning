from flask import Flask
from flask_cors import CORS
from flask_restful import Api
from controllers.ImageController import image_api
import os

app = Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
api = Api(app)


app.register_blueprint(image_api, url_prefix='/image')


@app.route('/', methods=['GET'])
def test():
  return "It works"


def create_folders():

    folders_upload=os.listdir(os.path.abspath("view/public/uploads"))
    
    if("Image_origin" not in folders_upload):
        os.mkdir(os.path.abspath("view/public/uploads/Image_origin"))
    
    if("Datasets" not in folders_upload):
        os.mkdir(os.path.abspath("view/public/uploads/Datasets"))

    if("Results" not in folders_upload):
        os.mkdir(os.path.abspath("view/public/uploads/Results"))
    
    if("References" not in folders_upload):
        os.mkdir(os.path.abspath("view/public/uploads/References"))
    
    if("Scribbles" not in folders_upload):
        os.mkdir(os.path.abspath("view/public/uploads/Scribbles"))



create_folders()

if __name__ == "__main__":
    app.run(threaded=True ,host="0.0.0.0", port=5000,debug=True)