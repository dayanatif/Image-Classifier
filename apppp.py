import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Streamlit app layout
st.title("Image Classifier App 📷")
st.write("Upload an image to classify it!")

# User input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    labels = decode_predictions(predictions)
    
    # Display the top prediction
    st.write("Predicted:", labels[0][0][1])  # Show class label
    st.write("Confidence:", round(labels[0][0][2] * 100, 2), "%")  # Show confidence percentage
