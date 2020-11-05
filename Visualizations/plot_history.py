# Adam Santos, BS Robotics Engineering
# 10/22/20
# Plots loss and accuracy for a given model's history
## history = model.fit(X, Y)

import matplotlib.pyplot as plt

def plot_acc_loss(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    axes = plt.gca()
    # axes.set_ylim([0, 1])
    plt.show()
