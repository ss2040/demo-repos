from flask import Flask, request, jsonify
from tensorflow import keras
import cv2
import numpy as np
import random
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from CV.source.data_preparation import haar_cascade
#model architecture
image_size=224
vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
for layer in vgg_conv.layers[:]:
    layer.trainable = False
model = Sequential()
model.add(vgg_conv)
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.load_weights("/home/ubuntu/Hackathon-D-Cube/CV/notebooks/vgg_64.h5")

app = Flask(__name__)

@app.route("/cv", methods=["PUT","POST","GET"])
def process_image():
    file = request.files['image']
    # Read the image via file.stream
    img = Image.open(file.stream).convert('RGB')
    img = np.array(img)

    #pre processing
    #face=haar_cascade.extract_face(img)
    img=cv2.resize(img,(224,224))
    pred=model.predict(np.expand_dims(img,axis=0))
    print(pred)
    result=np.argmax(pred)
    print(result)
    if result == 0:
        res="Drunk"
    else:
        res="Sober"
    score=pred[0][result]
    score=float(np.round(score,2))
    return jsonify({'state': res, 'confidence':score})


if __name__ == "__main__":
    app.run(debug=True)
