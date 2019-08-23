# import the necessary packages
import numpy as np
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.models import load_model

from keras.datasets import mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# reshape the data to four dimensions, due to the input of model
# reshape to be [samples][width][height][pixels]
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# normalization
x_train = x_train / 255.0
x_test = x_test / 255.0

# one-hot
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# parameters
EPOCHS = 10
INIT_LR = 1e-3
BS = 32
CLASS_NUM = 10
norm_size = 28


# define lenet model
def l_model(width, height, depth, NB_CLASS):
    model = Sequential()
    inputShape = (height, width, depth)
    # if we are using "channels last", update the input shape
    if K.image_data_format() == "channels_first":  # for tensorflow
        inputShape = (depth, height, width)
    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
 
    # softmax classifier
    model.add(Dense(NB_CLASS))
    model.add(Activation("softmax"))
 
    # return the constructed network architecture
    return model
model = l_model(width=norm_size, height=norm_size, depth=1, NB_CLASS=CLASS_NUM)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# Use generators to save memory
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit_generator(aug.flow(x_train, y_train, batch_size=BS),
                            steps_per_epoch=len(x_train) // BS,
                            epochs=EPOCHS, verbose=2)



# save json
from keras.models import model_from_json
json_string = model.to_json()
with open('m_lenet.json', 'w') as file:
    file.write(json_string)
# save weights
model.save_weights('m_weights.h5')



