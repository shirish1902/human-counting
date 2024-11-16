import streamlit as st
import os
import cv2
from ultralytics import YOLO
from PIL import Image
import tempfile

# Define YOLO model globally for reuse
model = YOLO('yolov8n-pose.pt')

# Define functions
def process_image_or_folder(input_path):
    if os.path.isfile(input_path):  # Single file
        model.predict(source=input_path, save=True, imgsz=320, conf=0.5)
        st.success("Processing complete for the file.")
    elif os.path.isdir(input_path):  # Directory
        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                filepath = os.path.join(input_path, filename)
                model.predict(source=filepath, save=True, imgsz=320, conf=0.5)
        st.success("Processing complete for all images in the folder.")
    else:
        st.error(f"Error: Invalid path '{input_path}'. Please provide a valid file or folder path.")

def process_video():
    video_path = 0  # Default to camera
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()  # Placeholder for displaying frames in Streamlit
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame)  # Inference without saving
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR")  # Display the frame
        else:
            break
    cap.release()

# Streamlit UI
st.title("YOLOv8 Pose Estimation App")

choice = st.radio("Choose an option:", ["Code 1 - Image/Folder Processing", "Code 2 - Real-time Video"])

if choice == "Code 1 - Image/Folder Processing":
    uploaded_file = st.file_uploader("Upload an image or zip containing images", type=["png", "jpg", "jpeg", "bmp", "zip"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".zip"):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_zip_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_zip_path, "wb") as f:
                    f.write(uploaded_file.read())
                st.info(f"Unzipping {uploaded_file.name}...")
                import zipfile
                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                process_image_or_folder(temp_dir)
        else:
            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            process_image_or_folder(temp_path)

elif choice == "Code 2 - Real-time Video":
    if st.button("Start Camera"):
        process_video()
