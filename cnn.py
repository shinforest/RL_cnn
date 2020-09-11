from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation, Conv2D, Flatten, Dense,Dropout
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
import time
import os
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import tensorflow as tf
import math

class CNN:
    def get_rewards():
        folder = os.listdir("assets")
        image_size = 50
        dense_size  = len(folder)

        X = []
        Y = []

        if ".DS_Store" in folder:
            folder.remove(".DS_Store") 

        for index, name in enumerate(folder):
            dir = "./assets/" + name
            print(dir)
            files = glob.glob(dir + "/*.jpg")
            for i, file in enumerate(files):
                image = Image.open(file)
                image = image.convert("RGB")
                image = image.resize((image_size, image_size))
                data = np.asarray(image)
                X.append(data)
                Y.append(index)

        X = np.array(X)
        Y = np.array(Y)
        X = X.astype('float32')
        X = X / 255.0

        Y = np_utils.to_categorical(Y, dense_size)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


        print(len(X_train))
        print(len(X_test))
        print(len(y_train))
        print(len(y_test))

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',input_shape=(50,50,3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(dense_size))
        model.add(Activation('softmax'))


        model.summary()

        optimizers ="adam"
        results = {}
        epochs = 100
        model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics=['accuracy'])
        history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs )


        return history.history['val_accuracy'][-1]


        # model_json_str = model.to_json()
        # open('mnist_mlp_model.json', 'w').write(model_json_str)
        # model.save_weights('mnist_mlp_weights.h5')


        # y_pred = np.argmax(model.predict(X_test), axis=-1)
        # print(y_pred)
        # print(len(y_pred))
