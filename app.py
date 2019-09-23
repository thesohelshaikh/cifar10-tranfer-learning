from flask import Flask, render_template, request

from scipy.misc import imread, imresize
from imageio import imwrite
import numpy as np
import keras.models
import re
import sys
import os
from load import *

# setting the path for our model
sys.path.append(os.path.abspath('./model'))

# Initialize the app
app = Flask(__name__)

# Default route
@app.route('/')
def hello():
    return 'Hello, world!'

if __name__ == '__main__':
    app.run()