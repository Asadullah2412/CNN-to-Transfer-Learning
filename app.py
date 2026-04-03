import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
# Load model
# model = tf.keras.models.load_model("models/model_efficient")
# model = tf.keras.models.load_model("models/model_mobilenet.keras")
model = tf.keras.models.load_model("models/model_efficient.keras")
st.set_page_config(page_title="Pneumonia Detector", layout="centered")

# Title
st.title("Pneumonia Detection System")
st.markdown("Upload a chest X-ray image to check for Pneumonia.")
# import sys
# st.write(sys.version)
import random

pneumonia_high = [
    "The model strongly indicates the presence of pneumonia-related patterns in the X-ray.",
    "Clear signs consistent with pneumonia have been detected in the image.",
    "The prediction suggests a high likelihood of pneumonia. Further medical evaluation is recommended."
]

pneumonia_medium = [
    "There are noticeable patterns that may indicate pneumonia.",
    "The model detects features that could be associated with pneumonia.",
    "Some abnormalities suggest a possible pneumonia condition."
]

pneumonia_low = [
    "There are weak signals that might be related to pneumonia, but the confidence is low.",
    "The model shows slight indications of pneumonia, but the result is uncertain.",
    "Possible pneumonia-related features detected, though not strongly conclusive."
]

normal_high = [
    "The X-ray appears normal with no visible signs of pneumonia.",
    "The model confidently classifies this scan as normal.",
    "No pneumonia-related abnormalities were detected."
]

normal_medium = [
    "The image mostly appears normal, with no strong signs of pneumonia.",
    "There are no clear indicators of pneumonia in this scan.",
    "The model suggests the image is likely normal."
]

normal_low = [
    "The scan appears mostly normal, but the confidence is not very high.",
    "No strong pneumonia indicators were found, though the result is slightly uncertain.",
    "The model leans toward a normal classification, but with limited confidence."
]

def generate_response(prediction):
    confidence = prediction if prediction > 0.5 else 1 - prediction

    if prediction > 0.5:  # Pneumonia
        if confidence > 0.85:
            return random.choice(pneumonia_high)
        elif confidence > 0.65:
            return random.choice(pneumonia_medium)
        else:
            return random.choice(pneumonia_low)

    else:  # Normal
        if confidence > 0.85:
            return random.choice(normal_high)
        elif confidence > 0.65:
            return random.choice(normal_medium)
        else:
            return random.choice(normal_low)
# Upload
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Uploaded Image", )

    if st.button("Predict"):
        # Preprocess
        image = Image.open(uploaded_file).convert("RGB")
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # ⚠️ IMPORTANT: match preprocessing to model

        img_array = preprocess_input(img_array)

        # Prediction
        prediction = model.predict(img_array)[0][0]
        response = generate_response(prediction)

        st.markdown("### Interpretation")
        st.write(response)
        # Output
        if prediction > 0.6:
            st.error(f"⚠️ Pneumonia Detected ({prediction:.2f})")
        else:
            st.success(f"✅ Normal ({1 - prediction:.2f})")
        