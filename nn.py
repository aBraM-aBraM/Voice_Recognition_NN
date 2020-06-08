import numpy as np
import keras


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

import librosa
import librosa.display



def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if(ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
        
    print(cm)
    
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def predict(model, test_batches):
    # TESTING
    test_imgs, test_labels = next(test_batches)
    plots(test_imgs, titles= test_labels)

    test_labels = test_labels[:,0]
    test_labels
    
    predictions = model.predict_generator(test_batches, steps=1, verbose=0)
    print(predictions)
    cm = confusion_matrix(test_labels, predictions[:,0])

    cm_plot_labels = ['0','1','2','3','4','5','6','7','8','9']
    #plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

def create_model():
    
    # creating the model
    
    # Sequential is a model with an array of layers
    model = Sequential()
    # 2D Convultional layer (output filters, dimensions of conv2d window, inputshape = (y,x,3=rgb))
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = (432, 288, 3)))
    model.add(MaxPooling2D(pool_size = (2,2)))
     # flattening the output from first layer
    model.add(Flatten())
    # Dense(number of output layers (num of different results), activation func)
    model.add(Dense(len(classes), activation='softmax'))

    # Adam optimizer
    # lr (learning rate)
    # loss (mistake calculation)
    model.compile(Adam(lr=.0001),loss='categorical_crossentropy',metrics=['accuracy']) 
    
    # Summary of model
    model.summary()
    
    return model
    
def load_model():
    print('a')
    return load_model('voice_recognition.h5')


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

imgs, labels = next(train_batches)

plots(imgs, titles=labels)


train_labels = []
train_samples = []

# labels will store an index of a result ranging from 0 to the amount of output neurons




# validation_steps= dataset / batch_size
validation_steps = 2000 / batch_size


model = load_model('voice_recognition.h5')
#model = create_model()
predict(model, test_batches)

# TRAIN




# saving the model entirely (architecture, weights, trained data)
#model.save('voice_recognition.h5')
# 




