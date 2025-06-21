import streamlit as st
import cv2
import numpy as np
import os
import time  # To generate a timestamp-based unique ID
from app.detect_faces import detect_faces  # Make sure your function is properly imported

# Create a unique folder for each session or user
def generate_unique_id():
    return str(int(time.time()))  # Use the current timestamp as a unique ID

st.title("👁️‍🗨️ Face Detection & Preprocessing")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Generate a unique ID for this session
    session_id = generate_unique_id()

    # Create a folder for this session in the outputs directory
    session_output_dir = os.path.join("outputs", session_id)
    os.makedirs(session_output_dir, exist_ok=True)

    # Read the image in OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Save temp image
    temp_image_path = os.path.join(session_output_dir, "temp_input.jpg")
    cv2.imwrite(temp_image_path, image)

    # Detect faces
    img_with_boxes, boxes, faces = detect_faces(temp_image_path)

    # Show original image with bounding boxes
    st.subheader("🔍 Faces Detected")
    st.image(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB), channels="RGB")

    # Save and show cropped faces
    if faces:
        st.subheader("🧠 Cropped Face(s)")

        # Save the cropped faces to the session-specific output folder
        for i, face in enumerate(faces):
            # Save cropped faces with a unique session ID in the filename
            face_filename = os.path.join(session_output_dir, f"cropped_face_{i+1}.jpg")
            cv2.imwrite(face_filename, face)
            st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), caption=f"Face {i+1}", channels="RGB")
    else:
        st.warning("No faces found.")
