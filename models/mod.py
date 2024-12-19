import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
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

# Output the result
print("Prediction probabilities:", prediction)

# Optional: Map probabilities to class names if available
# class_names = ['Class1', 'Class2', ...]  # Replace with your class labels
# predicted_class = class_names[np.argmax(prediction)]
# print("Predicted class:", predicted_class)
