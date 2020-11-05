"""
Adam Santos (addrick)
Utility to visualize each convolutional filter in a single network layer
Outputs an image maximizing filter's response for each filter in a given model layer
Uses gradient ascent to shift a generated gray image with randomized noise to maximize filter response
Reworked and expanded from Keras example code by [fchollet](https://twitter.com/fchollet)
Date modified: 10/31/20
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from Models import cifar10_modelfn as cf

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# The dimensions of our input image
img_width = 32
img_height = 32
# Our target layer: we will visualize the filters from this layer.
# See `model.summary()` for list of layer names, if you want to change this.
layer_index = 10
model = cf.load_weights()

# Set up a model that returns the activation values for our target layer
layer = model.get_layer(index=layer_index)
num_filters = layer.get_config()['filters']

feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)

"""
## Set up the gradient ascent process

The "loss" we will maximize is simply the mean of the activation of a specific filter in
our target layer. To avoid border effects, we exclude border pixels.
"""
def compute_loss(input_image, filter_index):
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)

"""
Our gradient ascent function simply computes the gradients of the loss above
with regard to the input image, and update the update image so as to move it
towards a state that will activate the target filter more strongly.
"""
# @tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img

"""
## Set up the end-to-end filter visualization loop

Our process is as follow:

- Start from a random image that is close to "all gray" (i.e. visually netural)
- Repeatedly apply the gradient ascent step function defined above
- Convert the resulting input image back to a displayable form, by normalizing it,
center-cropping it, and restricting it to the [0, 255] range.
"""
def initialize_image():
    # We start from a gray image with some random noise
    img = tf.random.uniform(shape=(1, img_width, img_height, 3), maxval=1)

    return img

def visualize_filter(filter_index):
    iterations = 100000
    learning_rate = 150.0
    img = initialize_image()
    for iteration in range(iterations):
        # learning_rate = learning_rate/1.2
        if iterations-iteration < iterations/1.5:
            learning_rate = learning_rate/10
            if iterations - iteration < iterations / 4:
                learning_rate = learning_rate/20
                if iterations - iteration < iterations / 8:
                    learning_rate = learning_rate/40
                    if iterations - iteration < iterations / 16:
                        learning_rate = learning_rate/80
                        if iterations - iteration < iterations / 32:
                            learning_rate = learning_rate / 160

        loss, img = gradient_ascent_step(img, filter_index, learning_rate)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img

def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15
    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)
    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img

"""
## Visualize the filters in the target layer:
Make a 8xn grid of the filters in the target layer
to get of feel for the range of different 
visual patterns that the model has learned.
"""
# Compute image inputs that maximize per-filter activations
# for the first 64 filters of our target layer
all_imgs = []
for filter_index in range(num_filters):
    print("Processing filter %d" % ((filter_index+1),))
    loss, img = visualize_filter(filter_index)
    plt.imshow(img)
    plt.show()
    all_imgs.append(img)

# Build a black picture with enough space for each filter in the layer
# Currently set to work with sets that have a multiple of 8 filters
margin = 1
n = 16
m = int(num_filters/n)
# n = np.int(np.sqrt(num_filters))
cropped_width = img_width
cropped_height = img_height
width = n * (cropped_width + margin)
height = m * (cropped_height + margin)
stitched_filters = np.zeros((height, width, 3))

# Fill the picture with our saved filters
for i in range(n):
    for j in range(m):
        img = all_imgs[i*m + j]
        stitched_filters[
            (cropped_height + margin) * j : (cropped_height + margin) * j + cropped_height,
            (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
            :,
        ] = img
keras.preprocessing.image.save_img("stitched_filter_" + str(layer_index) + ".png", stitched_filters)

