import os
import numpy as np
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model
import cv2 as cv

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
print("Training CNN classifier...")
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# TODO:
data_path = "test"
names = os.listdir(data_path)
input_res = (72, 128)
# load data from folders based on folder name:
images = []
labels = []
label_dict = {}
num_images = 0
for i, name in enumerate(names):
    # count examples
    image_files = os.listdir(data_path + '/' + name)
    num_images += len(image_files)
    im_path = data_path + '/' + names[0]
    # encoded category for name
    label_dict.update({name: i})

    # load image into example pool
    for im_name in image_files:
        im = data_path + '/' + name + '/' + im_name
        im = cv.imread(im)
        image = cv.resize(im, dsize=(input_res))
        images.append(image)
        labels.append(i)

# Normalize pixel values to be between 0 and 1
images = np.asarray(images)
images = images / 255.0
labels = np.asarray(labels)
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.33)
# Create the model
model = Sequential()
model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=(input_res[0], input_res[1], 3)))
model.add(Dropout(0.1))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(2048, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(10))
# Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
callbacks_list = []

history = model.fit(train_images, train_labels, batch_size=8, epochs=50,
                    validation_data=(test_images, test_labels), callbacks=callbacks_list)