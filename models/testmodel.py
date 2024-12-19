import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
# Load the model
# Load an image to be predicted
# path_AW='C:/Users/Md Ganim/Desktop/Program/AI_project/Final/models/image.png'
# img = image.load_img(path_AW, target_size=(224, 224))
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)

# model = tf.keras.models.load_model("C:/Users/Md Ganim/Desktop/Program/AI_project/Final/models/mobilenet_model1.h5")
# path_SB="C:/Users/Md Ganim/Desktop/Program/AI_project/Final/models/image.png"
# # Preprocess the input image
# img = image.load_img(path_SB, target_size=(224, 224))
# img = tf.keras.preprocessing.image.img_to_array(img)
# img = tf.keras.applications.mobilenet.preprocess_input(img)
# img = np.expand_dims(img, axis=0)
model = tf.keras.models.load_model('C:/Users/Md Ganim/Desktop/Program/AI_project/Final/models/mobilenet_model1.h5')

# Path to the input image
image_path = 'C:/Users/Md Ganim/Desktop/Program/AI_project/Final/models/image.png'  # Replace with your image path

# Preprocess the image
image = load_img(image_path, target_size=(224, 224))  # Resize to the model's input size
image_array = img_to_array(image)                     # Convert image to a numpy array
image_array = np.expand_dims(image_array, axis=0)     # Add batch dimension
image_array = preprocess_input(image_array)           # Apply preprocessing

# Predict using the model
prediction = model.predict(image_array)

# # Make predictions
# predictions = model.predict(img)

# Decode the predictions
predicted_class = tf.argmax(prediction[0])
