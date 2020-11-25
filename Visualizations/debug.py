from io import BytesIO

import requests
from PIL import Image
from keract import keract
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from Models import cifar10_modelfn as cf

# Check for GPUs and set them to dynamically grow memory as needed
# Avoids OOM from tensorflow greedily allocating GPU memory
model = cf.load_weights()

# url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_' \
#       'europeo4.jpg/250px-Gatto_europeo4.jpg'
# response = requests.get(url)
# image = Image.open(BytesIO(response.content))
# image = image.crop((0, 0, 224, 224))

image = Image.open('Visualizations/dog10.png')

image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
# yhat = model.predict(image)
# label = decode_predictions(yhat)
# label = label[0][0]
# print('{} ({})'.format(label[1], label[2] * 100))
#
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
activations = keract.get_activations(model, image)
first = activations.get('block1_conv1')
keract.display_activations(activations, save=True)

