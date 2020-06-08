import numpy as np
import keras
import nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

import itertools
import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy


import librosa.display


# declare paths
train_path = 'train'
valid_path = 'valid'
test_path = 'test'

classes = ['0','1','2','3','4','5','6','7','8','9']
target_size = (432,288)
batch_size=10

# creating batches
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=target_size,classes=classes, batch_size=batch_size)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=target_size,classes=classes, batch_size=batch_size)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=target_size,classes=classes, batch_size=batch_size)


model = nn.load_model()

# TESTING
test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles= test_labels)

test_labels = test_labels[:,0]
test_labels

predictions = model.predict_generator(test_batches, steps=1, verbose=0)
print(predictions)