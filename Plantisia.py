import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Define the path to the model file
MODEL_PATH = "trained_model.keras"

# TensorFlow Model Prediction
def model_prediction(test_image):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

    try:
        # Load image
        image = Image.open(test_image).convert('RGB')
        image = image.resize((256, 256))  # Ensure image is the correct size
        input_arr = np.array(image)  # Convert image to numpy array
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

        # Normalize the image
        input_arr = input_arr / 255.0  # Adjust normalization if needed

        # Make prediction
        predictions = model.predict(input_arr)
        return np.argmax(predictions), np.max(predictions)  # Return index of max element and probability
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Inject custom CSS for background image
st.markdown("""
    <style>
    .main {
        background-image: url('home_page.jpg'); /* Use the correct path here */
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition", "Information", "FAQ"])

# Main Page
if app_mode == "Home":
    st.title("🌿 PLANTISIA 🌿")
    
    # Ensure that 'home_page.jpg' is in the same directory or provide the correct path
    st.image("home_page.jpg", use_column_width=True)
    
    st.markdown("""
    Welcome to the PLantisia! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.title("Disease Recognition")
    st.write("Upload an image of a plant leaf to identify potential diseases.")
    
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        # Display the uploaded image
        st.image(test_image, caption='Uploaded Image', use_column_width=True)
        
        # Predict button
        if st.button("Predict"):
            st.spinner("Analyzing the image...")
            result_index, probability = model_prediction(test_image)
            if result_index is not None:
                # Reading Labels
                class_names = [
                    'Tomato___Late_blight',
                    'Tomato___healthy',
                    'Grape___healthy',
                    'Potato___healthy',
                    'Corn_(maize)___Northern_Leaf_Blight',
                    'Tomato___Early_blight',
                    'Tomato___Septoria_leaf_spot',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'Strawberry___Leaf_scorch',
                    'Peach___healthy',
                    'Coffee__Rust',
                    'Apple___Apple_scab',
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                    'Tomato___Bacterial_spot',
                    'Coffee__red spider mite',
                    'Apple___Black_rot',
                    'Cherry_(including_sour)___Powdery_mildew',
                    'Peach___Bacterial_spot',
                    'Apple___Cedar_apple_rust',
                    'Tomato___Target_Spot',
                    'Chili__whitefly',
                    'Pepper,_bell___healthy',
                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                    'Potato___Late_blight',
                    'Chili__healthy',
                    'Tomato___Tomato_mosaic_virus',
                    'Strawberry___healthy',
                    'Apple___healthy',
                    'Grape___Black_rot',
                    'Chili__leaf spot',
                    'Potato___Early_blight',
                    'Cherry_(including_sour)___healthy',
                    'Coffee__healthy',
                    'Corn_(maize)___Common_rust',
                    'Grape___Esca_(Black_Measles)',
                    'Tomato___Leaf_Mold',
                    'Chili__yellowish',
                    'Chili__leaf curl',
                    'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Pepper,_bell___Bacterial_spot',
                    'Corn_(maize)___healthy'
                ]
                st.success(f"**Prediction:** The model predicts this image is most likely: **{class_names[result_index]}** with a probability of **{probability:.2f}**.")

# Information Page
elif app_mode == "Information":
    st.title("Information")
    st.write("""
    ### About the Plant Disease Recognition System

    Our Plant Disease Recognition System is a cutting-edge tool designed to identify plant diseases from images. Leveraging advanced machine learning algorithms, the system provides accurate and timely disease detection to help farmers and gardeners take prompt action.

    **Key Features:**
    - **High Accuracy:** Utilizes state-of-the-art deep learning models trained on a diverse dataset of plant images.
    - **User-Friendly Interface:** Easy-to-use interface for quick and efficient disease identification.
    - **Fast Processing:** Provides results within seconds for rapid decision-making.

    


    """)

# FAQ Page
elif app_mode == "FAQ":
    st.title("FAQ")
    st.write("""
    ### Frequently Asked Questions (FAQ)

    **Q1: How does the Plant Disease Recognition System work?**
    - **A1:** The system uses deep learning algorithms to analyze images of plant leaves and identify potential diseases. Simply upload an image, and the system will process it to provide a prediction.

    **Q2: What types of plant diseases can the system detect?**
    - **A2:** The system can identify a variety of plant diseases based on the trained model. The specific diseases include late blight, early blight, bacterial spot, and more.

    **Q3: How accurate is the disease prediction?**
    - **A3:** The accuracy of the prediction depends on the quality of the uploaded image and the trained model. Our system aims to provide high accuracy, but results should be verified with a plant specialist for critical decisions.

    **Q4: What should I do if the system fails to recognize a disease?**
    - **A4:** If the system does not recognize a disease, ensure that the image is clear and properly represents the plant leaf. For further assistance, you may consult a plant pathologist or use additional diagnostic tools.

    **Q5: Is my data safe when using the system?**
    - **A5:** Yes, we prioritize user privacy and data security. The images uploaded for analysis are not stored and are used solely for the purpose of providing predictions.

    **Q6: Can I use the system on mobile devices?**
    - **A6:** Yes, the system is designed to be accessible from various devices, including mobile phones and tablets. Ensure you have a stable internet connection for the best experience.
    """)

