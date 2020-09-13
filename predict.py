# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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

graph = tf.get_default_graph()


class Predictor:

    def predict(self,model,episode,image_size=50):
        global graph
        with graph.as_default():

            X = []
            image = Image.open("test_assets/{}/target.jpg".format(episode))
            image = image.convert("RGB")
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X.append(data)

            X = np.array(X)
            X = X.astype('float32')
            X = X / 255.0
            print(X.shape)
            l=model.predict(X)
            return(l[0][0])
