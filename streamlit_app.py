import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Streamlit App
st.title("Webcam Image Capture")

# Webcam feed
camera = cv2.VideoCapture(0)  # 0 for the default camera
stframe = st.empty()  # Placeholder for the video feed

# Placeholder to store captured image
captured_image = None

while True:
    ret, frame = camera.read()
    if not ret:
        st.error("Unable to access the camera.")
        break

    # Convert BGR to RGB for display in Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    stframe.image(frame_rgb, caption="Live Camera Feed", channels="RGB")

    # Button to capture the image
    if st.button("Capture Image"):
        captured_image = frame_rgb
        break

camera.release()

if captured_image is not None:
    # Display captured image
    st.image(captured_image, caption="Captured Image", channels="RGB")

    # Save the captured image locally
    save_path = "captured_image.jpg"
    cv2.imwrite(save_path, cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR))
    st.success(f"Image saved as {save_path}")
