
##  Face Cloaking System (PGD-Based Adversarial Protection)

This project implements a **privacy-preserving facial cloaking system** that protects individuals from facial recognition by generating **adversarial noise** targeted at deep face embedding models like FaceNet. The system uses **PGD (Projected Gradient Descent)** to perturb facial embeddings, making them unrecognizable to modern facial recognition systems, while keeping the image human-recognizable.

---

##  Key Features

*  **Targeted Cloaking**: Minimizes similarity between original and perturbed embeddings.
*  **Face Embedding Attack**: Uses PGD on InceptionResnetV1 (FaceNet) embeddings.
*  **Image & Face Handling**: Automatically detects faces and applies cloaking to each.
*  **Similarity Evaluation**: Tests cosine similarity between original and cloaked face.
*  **Visual Blending**: Optionally blends cloaked faces back into original image.

---


##  Requirements

Install dependencies using:

```bash
pip install torch torchvision facenet-pytorch opencv-python numpy
```

Ensure your environment has:

* Python ≥ 3.8
* PyTorch (CPU or GPU)
* OpenCV (cv2)
* facenet-pytorch

---

##  Usage

### 1. Cloak Faces in an Image

```python
from apply_cloak_to_faces import apply_cloak_to_faces
from detect_faces import detect_faces
import cv2
import os

# Load and detect
image_path = 'static/input.jpg'
original = cv2.imread(image_path)
boxes, faces = detect_faces(original)

# Apply cloaking
output = apply_cloak_to_faces(original, boxes, faces, session_folder="static/")

# Save result
cv2.imwrite("static/cloaked_output.jpg", output)
```

### 2. Test Cosine Similarity

Compare embedding similarity of original vs. cloaked face:

```bash
python test_similarity.py
```

Inside `test_similarity.py`:

```python
original = r"C:\path\to\original.jpg"
cloaked = r"C:\path\to\cloaked.jpg"
```

---

##  How It Works

* The system loads a pre-trained FaceNet  model.
* Each detected face is converted to a tensor and passed through PGD attack:
  * A loss function maximizes the **cosine distance** between embeddings.
  * Perturbations are constrained by an `epsilon` bound and applied iteratively.
* Final perturbed face is blended into the original image to preserve appearance.

---

## ⚙ Parameters

You can tweak the strength and behavior of the attack:

| Parameter | Description                  | Recommended Value |
| --------- | ---------------------------- | ----------------- |
| `epsilon` | Maximum allowed perturbation | `0.3` to `0.5`    |
| `alpha`   | Step size per iteration      | `0.01` to `0.03`  |
| `iters`   | Number of PGD iterations     | `30` to `50`      |

---

## Why This Matters

This tool demonstrates the **vulnerability of facial recognition systems** and helps build **user-controlled privacy tools**. Adversarial cloaking is a proactive defense against:

* Mass surveillance
* Unauthorized facial scraping
* Deepface tracking

---

##  To-Do / Extensions

[ ] Add CW and BIM attacks
[ ] Compression robustness testing (JPEG, MP4)
[ ] Web UI using Streamlit
[ ] Batch processing for videos

---

## License

This project is for academic and ethical use only. Do not use for any unauthorized tampering with facial recognition systems.

---

## Acknowledgements

* [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
* PGD Attack: Madry et al. (2017)
