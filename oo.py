import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

# Load the pre-trained VGG16 model
model = tf.keras.applications.VGG16(weights='imagenet')

# Load an image of a cat
img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Use the model to predict the class of the image
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=1)[0])

# Load an image of a dog
img_path = 'dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Use the model to predict the class of the image
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=1)[0])