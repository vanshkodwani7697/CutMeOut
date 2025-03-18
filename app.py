import streamlit as st
import numpy as np
import requests
import os
import tensorflow as tf
from PIL import Image
import io

# Model path and Hugging Face model URL
MODEL_PATH = "person_segmentation_Unet_Resnet50.keras"
MODEL_URL = "https://huggingface.co/vanshkodwani7697/CutMeOut/tree/main"

# Function to download model if not available
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model... Please wait.")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            st.success("Model downloaded successfully!")
        else:
            st.error("Failed to download model. Check the URL or internet connection.")
            return None
    return tf.keras.models.load_model(MODEL_PATH)

# Load the model
model = download_model()
if model is None:
    st.stop()  # Stop execution if the model couldn't be loaded

# Function to preprocess image
def preprocess_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image = image.resize((256, 256))  # Resize to match model input size
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to get prediction
def get_prediction(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    mask = np.squeeze(prediction)  # Remove batch and channel dimensions
    return mask

# Streamlit UI
st.title("CutMeOut - Human Mask Segmentation")
st.header("Upload an Image for Prediction")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Get prediction
    mask = get_prediction(image)

    if mask is not None:
        st.image(mask, caption="Segmentation Mask", use_container_width=True)
    else:
        st.error("Error in generating segmentation mask.")
