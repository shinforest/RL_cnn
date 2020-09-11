import tensorflow as tf
from keras.utils import np_utils
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation, Conv2D, Flatten, Dense,Dropout
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
import math
import os
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
from keras.models import Sequential, model_from_json



class Predictor:

    def predict(self,model,episode,image_size=50):
        X = []
        files = glob.glob("test_assets/{}/*.jpg".format(episode))
        image = Image.open(files[0])
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)

        X = np.array(X)
        X = X.astype('float32')
        X = X / 255.0
        l=model.predict(X)
        return(l[0][0])
