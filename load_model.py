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

class Model:
    def load_model(self):
        self.model2 = model_from_json(open("models/coral_model/mnist_mlp_model.json", 'r').read())
        self.model2.load_weights("models/coral_model/mnist_mlp_weights.h5")
        self.model2.summary()
        return self.model2
