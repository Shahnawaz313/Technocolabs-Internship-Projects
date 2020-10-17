import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, Response, request
from tensorflow.keras.models import model_from_json

json = open('json_model.json', 'r')
load_json = json.read()
json.close()

model = model_from_json(load_json)
model.load_weights('weights.h5')

labels = list('ABC')

def image_predict(image):
    return labels[np.argmax(model.predict(image))]





app = Flask(__name__)

@app.route("/", methods = ['GET','POST'])


def index():
    pred = None
    message = None
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.convert('L')
                img = img.resize((50,50))
                img = np.asarray(img)
                print(img.shape)
                img = img.reshape((1,50,50,1))
                img = img/255.0
                pred = image_predict(img)
        except:
            message = "Please upload an Image"
            return render_template('index.html', message = message)
    return render_template("index.html", pred = pred, message = message)




    


if __name__=='__main__':
    app.run(debug=True)
