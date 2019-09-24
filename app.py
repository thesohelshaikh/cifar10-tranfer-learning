from flask import Flask, render_template, request, jsonify

from scipy.misc import imread, imresize
from imageio import imwrite
import numpy as np
import keras.models
import re
import sys
import os

# Dependencies
import base64
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array


# Initialize the app
app = Flask(__name__)

def get_model():
	global model
	model = load_model('./model/cifar10_ResNet20v1_model.h5')
	model._make_predict_function()
	print('Model Loaded!')

def preprocess_image(image, target_size):
	if image.mode != 'RGB':
		image = image.convert('RGB')
	image = image.resize(target_size)
	image = img_to_array(image)

	img = image.astype('float32')
	img /= 255
	c = np.zeros(32*32*3).reshape((1,32,32,3))
	c[0] = img
	return c
	# image = np.expand_dims(image, axis=0)
	# return image

print('Loading model..............') 
get_model()

@app.route('/predict', methods=['POST'])
def predict():
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))
	processed_image = preprocess_image(image, target_size=(32,32))

	prediction = model.predict(processed_image).tolist()

	cifar10_labels = np.array([
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'])

	bestnum = 0.0
	bestclass = 0
	for n in [0,1,2,3,4,5,6,7,8,9]:
		if bestnum < prediction[0][n]:
			bestnum = prediction[0][n]
			bestclass = n
	
	predicted_class = cifar10_labels[bestclass]
	print(prediction)
	print(predicted_class)
	response = {
		'predicted_class' : predicted_class
	}

	return jsonify(response)

@app.route('/')
def hello():
	return "Hello"

@app.route('/hello', methods = ['POST'])
def hi():
	message = request.get_json(force=True)
	name = message['name']
	response = {
		'greeting' : 'Hello, ' + name + '!'
	}
	return jsonify(response)

if __name__ == '__main__':
    app.run()