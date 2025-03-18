import streamlit as st
import numpy as np
import requests
from PIL import Image
import io

# Hugging Face API endpoint
API_URL = "https://huggingface.co/vanshkodwani7697/CutMeOut/tree/main"

# Set headers with your Hugging Face token (optional if the model is public)
HEADERS = {}

# Function to preprocess image
def preprocess_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image = image.resize((256, 256))  # Resize to match model input size
    image_array = np.array(image) / 255.0  # Normalize the image
    return image_array

# Function to send image to Hugging Face API
def get_prediction(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()

    response = requests.post(API_URL, headers=HEADERS, files={"file": image_bytes})

    if response.status_code == 200:
        return np.array(response.json())  # Ensure the response is converted properly
    else:
        return None

# Streamlit UI
st.title("CutMeOut - Human Mask Segmentation")
st.header("Upload an Image for Prediction")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and get prediction
    mask = get_prediction(image)

    if mask is not None:
        st.image(mask, caption="Segmentation Mask", use_container_width=True)
    else:
        st.error("Error in fetching prediction from Hugging Face API")
