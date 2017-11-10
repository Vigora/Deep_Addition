import keras
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam

import tensorflow as tf
import numpy as np

print(keras.__version__)
print(tf.__version__)

len = 500000
len1 = 100000

x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train = x_train.reshape(len, 6)
x_test = x_test.reshape(len1, 6)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 201
x_test /= 201
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')

n_classes = 201
# convert class vectors to binary One Hot Encoded
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)
print(y_train[0])

# Training Parameters for basic MNIST
learning_rate = 0.1
training_epochs = 30
batch_size = 64

# Network Parameters
n_input = 6 #
n_hidden_1 = 101 # 1st layer number of neurons
n_hidden_2 = 201 # 2nd layer number of neurons
n_classes = 201 # MNIST classes for prediction(digits 0-9 )

model = Sequential()
model.add(Dense(n_hidden_1,  input_shape=(n_input,), name = "Dense_1"))
model.add(Activation('relu', name = "Relu1"))
model.add(Dense(n_hidden_2, name = "Dense_2"))
model.add(Activation('relu', name = "Relu2"))
model.add(Dense(n_classes, name = "Output"))
model.add(Activation('softmax', name = "Softmax_output"))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='ADAM', #optimizer='SGD',
              metrics=['accuracy'])



history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=training_epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('my_model_50.h5')

''' for dataset
exclude = [ i for i in range(0,101,2)]
exclude1 = []
x_train, y_train = generator(len,exclude)
x_test, y_test = generator(len1,exclude1)
model.save('my_model.h5')
Test accuracy: 0.37375

exclude = [50,51,52]
# exclude = [ i for i in range(0,101,2)]
exclude1 = []
model.save('my_model_50.h5')
Test accuracy: 0.61858 for 10 epochs
Test accuracy: loss: 0.9530 - acc: 0.7978 - val_loss: 0.9124 - val_acc: 0.8045 for 20 epochs

exclude = [50]
exclude1 = []
Test accuracy: loss: 0.9504 - acc: 0.8268 - val_loss: 0.9363 - val_acc: 0.8201 for 20 epochs


exclude = [50,51,52]
exclude1 = []
Test accuracy: loss: 1.4888 - acc: 0.8489 - val_loss: 1.5593 - val_acc: 0.8368 for 30 epochs

'''


""""
import h5py
import numpy as np

filename = 'C:/Users/Rommel/Downloads/dl_dev_course-master/dl_dev_course-master/week01/addition-model_9980.hd5'
f = h5py.File(filename, 'r')

f.name
f.keys()

# List all groups
print("Keys: %s" % f.keys())
a_group_key = f.keys() # [0]

# Get the data
data = list(f[a_group_key])
"""