"""
 Simple CNN model for the CIFAR-10 Dataset
 @author: Adam Santos
"""
import numpy
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model

def train(save_best=True):
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    print("Training small CIFAR10 CNN classifier...")
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    # load data
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Create the model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3), kernel_constraint=maxnorm(4)))
    model.add(Dropout(0.1))
    # model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.1))
    # model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.1))
    # model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(10, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.4))
    # model.add(Dense(2048, activation='relu', kernel_constraint=maxnorm(3)))
    # model.add(Dropout(0.5))
    # model.add(Dense(2048, activation='relu', kernel_constraint=maxnorm(3)))
    # model.add(Dropout(0.5))
    model.add(Dense(10))
    # Compile model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    callbacks_list = []
    if save_best:
        filepath = "best_cifar_cnn_weights_no_pooling.hdf5"
        # filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list.append(checkpoint)
    history = model.fit(train_images, train_labels, batch_size=64, epochs=500,
                        validation_data=(test_images, test_labels), callbacks=callbacks_list)
    return [model, history]

def load_weights():
    # load YAML and create model
    # yaml_file = open('model.yaml', 'r')
    # loaded_model_yaml = yaml_file.read()
    # yaml_file.close()
    # loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model = load_model("best_cifar_cnn_weights.hdf5")
    print("Loaded model from disk")
    loaded_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return loaded_model

def eval(model):
    # load data
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    score = model.evaluate(test_images, test_labels, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
