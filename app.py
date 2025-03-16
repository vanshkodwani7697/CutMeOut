import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Define the custom metric
def iou_metric(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / union

# Register the custom metric
custom_objects = {'iou_metric': iou_metric}

# Load the trained model
model = tf.keras.models.load_model(r"C:\\Users\\vansh\\Downloads\\project\\person_segmentation_Unet_Resnet50.keras", 
                                   custom_objects=custom_objects)

# Function to preprocess image
def preprocess_image(image):
    # Convert RGBA to RGB if the image has 4 channels
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image = image.resize((256, 256))  # Resize to match model input size
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit UI
st.title("CutMe"
"Out- Human Mask Segmentation")
st.header("Upload an Image for Prediction")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(preprocessed_image)

    # If the output is a segmentation mask, squeeze and display it
    if prediction.shape[-1] == 1:
        mask = np.squeeze(prediction)  # Remove the channel dimension
        st.image(mask, caption="Segmentation Mask", use_column_width=True)
    else:
        st.write("### Prediction:", prediction)
