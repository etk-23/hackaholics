import cv2
import numpy as np

def apply_basic_cloak(input_path, output_path, epsilon=50):
    img = cv2.imread(input_path)
    noise = np.random.uniform(-epsilon, epsilon, img.shape).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, noisy_img)
    print(f"✅ Cloaked image saved to: {output_path}")

# Use it
apply_basic_cloak("../outputs/face_0.jpg", "cloak_face.jpg")
