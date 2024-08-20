import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import cv2

# Load the model
model = tf.keras.models.load_model('trained_model.keras')
model.summary()

# Path to the image
image_path = r"C:\Users\91600\OneDrive\Desktop\Internship project\test\Apple___Apple_scab\0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG"

# Read and process the image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image
plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.title("Test Image")
plt.xticks([])
plt.yticks([])
plt.show()

# Prepare the image for prediction
target_size = (256, 256)  # Ensure this matches the training data
image = load_img(image_path, target_size=target_size)
input_arr = img_to_array(image)
input_arr = np.expand_dims(input_arr, axis=0) 
input_arr = input_arr / 255.0  # Normalize to [0, 1]

print(f"Input array shape: {input_arr.shape}")

# Make prediction
prediction = model.predict(input_arr)
print(f"Raw prediction probabilities: {prediction}")

result_index = np.argmax(prediction)
print(f"Predicted class index: {result_index}")

# Class names list
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Cherry_(including_sour)___healthy', 
    'Cherry_(including_sour)___Powdery_mildew', 'Chili__healthy',
    'Chili__leaf curl', 'Chili__leaf spot', 'Chili__whitefly', 'Chili__yellowish',
    'Coffee__healthy', 'Coffee__red spider mite', 'Coffee__Rust', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy',
    'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Peach___Bacterial_spot', 
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 
    'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# Display the result
model_prediction = class_names[result_index]
plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.title(f"Disease: {model_prediction}")
plt.xticks([])
plt.yticks([])
plt.show()

print(f"Predicted class: {model_prediction}")
