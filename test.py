# !pip install tensorflow-datasets

import tensorflow as tf


# Helper libraries
import math
import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Activation, Flatten, Conv2D, MaxPooling2D
X_train = np.array([[-1.0,-1.0], [1.0, -1.0], [1.0,1.0]])
y_train = np.array([[1.0], [-1.0], [0.0]])

X_test = np.array([[1.0,-1.0], [1.0, -1.0]])
y_test = np.array([[1.0], [-1.0]])

features_count = 2

model = Sequential([
    Conv1D(filters=256, kernel_size=5, padding='same', activation='relu', input_shape=(1,features_count)),
    MaxPooling2D,
    Flatten(),
    Dense(64),
    Activation('relu'),
    Dense(32),
    Activation('softmax'),
    Dense(1),
])

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

model.compile(optimizer="adam", loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=1)
model.evaluate(X_test, y_test, batch_size=1)