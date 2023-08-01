from flask import Flask, render_template, request, url_for
import os
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename
import cv2

model = load_model('./predictor')

app = Flask(__name__, static_folder='static')

def detect_person_in_image(file_path):
    # Load the pre-trained SSD model for object detection
    model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(model_path)

    # Read the image
    img = cv2.imread(file_path)
    # Convert the image to grayscale for faster processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 1:
        return True
    else:
        return False

@app.route('/')
def mainfun():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predictor():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename)
        )
        imgFilePath = os.path.join('static', 'imgUploads', secure_filename(f.filename))
        f.save(imgFilePath)

        # Check if there is a person in the image
        if not detect_person_in_image(file_path):
            result = "No person found in the image."
            return render_template('index.html', result=result, fname=imgFilePath)

        img = keras.utils.load_img(file_path, target_size=(64, 64))
        x = keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=1)
        print(classes[0])
        if classes[0] > 0.5:
            result = "female"
        else:
            result = "male"

        return render_template('index.html', result=result, fname=imgFilePath)

if __name__ == "__main__":
    app.run(debug=True)
