# import the necessary packages
import os
import argparse

from keras.models import load_model
from keras.preprocessing.image import img_to_array
import PIL
from PIL import Image
import numpy as np
import flask
import io

import src.utils.utils as ut
import src.data.crop as cr

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None


def load_pretrained_model():
    # load the pretrained model
    global model
    load_model_from = os.path.join(ut.dirs.model_dir, 'vgg16' + '.h5')

    model = load_model(load_model_from)
    model._make_predict_function()


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    wi = ut.params.cropwidth
    if(np.array(image.getdata()).astype(np.float32).shape[0] > wi**2):
        image = cr.crop_image(image)

    # resize the input image and preprocess it
    if (type(image) is not PIL.JpegImagePlugin.JpegImageFile):
        img = Image.fromarray(image)
    else:
        img = image
    image = img.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image/255.

    # return the processed image
    return image


def interprete_prediction(prediction):
    class_dict = {0: 'bad', 1: 'good'}
    rp = int(round(prediction[0][0]))
    return float(prediction[0][0]), rp, class_dict.get(rp)

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            imagename = flask.request.files["image"].filename
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(ut.params.image_width,
                                                 ut.params.image_heigth))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)

            (prob, classcode, classname) = interprete_prediction(preds)

            data["predictions"] = []
            # loop over the results and add them to the list of
            # returned predictions
            r = {"label": classcode,
                 "name": classname,
                 "p_good": prob,
                 "p_bad": 1-prob,
                 "file": imagename}
            data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))

    parser = argparse.ArgumentParser(description='Run the classifier API')
    parser.add_argument('--d', type=int, default=0,
                        help='debbuging mode\n\
                         0: False, 1: True (default: 0)')
    args = parser.parse_args()

    load_pretrained_model()

    debug_mode = False
    if(args.d == 1):
        debug_mode = True

    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
