import sys
import os
import cv2

# Print current sys.path to check where Python is looking
print("Current sys.path:")
print(sys.path)

# Force Python to find the 'app' folder
app_path = os.path.join(os.path.dirname(__file__), "app")
print(f"Adding app path: {app_path}")
sys.path.append(app_path)  # Add 'app/' folder to sys.path

# Verify if app is in sys.path
print("Updated sys.path:")
print(sys.path)

# Now try importing detect_faces
try:
    from detect_faces import detect_faces
    print("✅ Successfully imported detect_faces!")
except ImportError as e:
    print("❌ ERROR: Failed to import detect_faces.")
    print(f"Error details: {e}")
    sys.exit(1)

# If import is successful, the rest of the code can run

# Image path to test
image_path = "test_images/test1.jpg"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Run face detection
img, boxes, faces = detect_faces(image_path)

# Save each detected face as an image
for i, face in enumerate(faces):
    cv2.imwrite(f"{output_dir}/face_{i}.jpg", face)

# Draw bounding boxes on the original image
for (x1, y1, x2, y2) in boxes:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save the final image with boxes
cv2.imwrite(f"{output_dir}/image_with_boxes.jpg", img)

print(f"✅ Done: {len(faces)} face(s) detected. Output saved to /outputs")
