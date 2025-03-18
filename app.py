import streamlit as st
import numpy as np
import requests
import os
import tensorflow as tf
from PIL import Image

# Define model path and Hugging Face model URL
MODEL_PATH = "person_segmentation_Unet_Resnet50.keras"
MODEL_URL = "https://huggingface.co/vanshkodwani7697/CutMeOut/tree/main"

# Custom metric (if needed)
def iou_metric(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7)

# Function to download model safely
def download_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        st.info("Downloading model... Please wait.")
        response = requests.get(MODEL_URL, stream=True)

        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)

            if os.path.getsize(MODEL_PATH) > 0:
                st.success("Model downloaded successfully!")
            else:
                st.error("Downloaded file is empty. Try again.")
                return None
        else:
            st.error(f"Failed to download model. HTTP Status: {response.status_code}")
            return None

    try:
        custom_objects = {"iou_metric": iou_metric}  # Register custom metrics
        return tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = download_model()
if model is None:
    st.stop()

# Function to preprocess image
def preprocess_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)

# Function to get prediction
def get_prediction(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return np.squeeze(prediction)

# Streamlit UI
st.title("CutMeOut - Human Mask Segmentation")
st.header("Upload an Image for Prediction")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    mask = get_prediction(image)

    if mask is not None:
        st.image(mask, caption="Segmentation Mask", use_container_width=True)
    else:
        st.error("Error generating segmentation mask.")
