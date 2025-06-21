import cv2  # OpenCV for image processing
import mediapipe as mp  # Pretrained ML models (we're using face detection)
import numpy as np  # For handling image arrays

# Load the face detection module from MediaPipe
mp_face_detection = mp.solutions.face_detection

def detect_faces(image_path):
    """
    This function takes an image path, detects all faces in it,
    crops and resizes each face, and returns the results.
    """
    
    # Load the image from the file path
    image = cv2.imread(image_path)
    h, w, _ = image.shape  # Get image dimensions

    face_crops = []  # Store cropped face images
    boxes = []       # Store bounding boxes (to draw on image later)

    # Create the face detection model
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        # Convert image from BGR (OpenCV) to RGB (MediaPipe requirement)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run face detection
        results = detector.process(rgb_image)

        # If faces are found
        if results.detections:
            for det in results.detections:
                # Get relative bounding box (values between 0–1)
                bbox = det.location_data.relative_bounding_box

                # Convert relative coordinates to actual pixel positions
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = x1 + int(bbox.width * w)
                y2 = y1 + int(bbox.height * h)

                # Ensure box is within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                # Crop the face from the original image
                face_crop = image[y1:y2, x1:x2]

                # Resize face to 160x160 (standard input for FaceNet etc.)
                face_crop = cv2.resize(face_crop, (160, 160))

                # Store the cropped face and bounding box
                face_crops.append(face_crop)
                boxes.append((x1, y1, x2, y2))

    # Return everything: original image, boxes, and face crops
    return image, boxes, face_crops
