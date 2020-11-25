"""
 Simple CNN model for the CIFAR-10 Dataset
 @author: Adam Santos
"""
import numpy
from keras.constraints import maxnorm
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Convolution2D
import tensorflow as tf
from keras.utils import np_utils
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

    print("Training MNIST CNN classifier...")
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # reshape to be [samples][width][height][channels]
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    # normalize inputs from 0-255 to 0-1
    train_images = X_train / 255
    test_images = X_test / 255
    # one hot encode outputs
    train_labels = np_utils.to_categorical(y_train)
    test_labels = np_utils.to_categorical(y_test)

    # Create the model
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=3, padding='valid', input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, kernel_size=5, padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    callbacks_list = []
    if save_best:
        filepath = "best_mnist_cnn_weights.hdf5"
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
    loaded_model = load_model("best_mnist_cnn_weights.hdf5")
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


