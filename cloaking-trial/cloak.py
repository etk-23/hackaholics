import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from torchvision.transforms.functional import to_pil_image
import cv2

# Load the pretrained FaceNet model
model = InceptionResnetV1 (pretrained='vggface2').eval()

# Load and preprocess image
def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # shape: [1, 3, 160, 160]

# Save tensor image
def save_image(tensor_img, path):
    img = tensor_img.squeeze().permute(1, 2, 0).detach().numpy()
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)

# PGD attack function
def pgd_cloak_image(input_path, output_path, epsilon = 0.037 , alpha = 0.002, iters = 16):
    image = load_image(input_path).clone().detach()
    orig_image = image.clone().detach()
    perturbed = image.clone().detach().requires_grad_(True)

    # Strong fake target embedding
    target_embedding = model(orig_image).detach() + torch.randn_like(model(orig_image)) * 5

    loss_fn = torch.nn.MSELoss()

    for i in range(iters):
        output = model(perturbed)
        loss = loss_fn(output, target_embedding)
        loss.backward()

        with torch.no_grad():
            grad = perturbed.grad.sign()
            perturbed = perturbed + alpha * grad
            perturbed = torch.max(torch.min(perturbed, orig_image + epsilon), orig_image - epsilon)
            perturbed = torch.clamp(perturbed, 0, 1)

        perturbed.requires_grad = True  # reset for next iteration

    # Save result
    save_image(perturbed, output_path)
    print(f"✅ PGD cloaked image saved to: {output_path}")


# Run it
pgd_cloak_image("../outputs/face_0.jpg", "cloak_face.jpg", epsilon = 0.037 , alpha = 0.002, iters = 16)
# Compare embeddings function
def compare_embeddings(path1, path2):
    img1 = load_image(path1)
    img2 = load_image(path2)

    with torch.no_grad():
        emb1 = model(img1)
        emb2 = model(img2)

    similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
    distance = torch.norm(emb1 - emb2)

    print(f"\n🔍 Cosine Similarity: {similarity.item():.4f}")
    print(f"📏 Euclidean Distance: {distance.item():.4f}")
compare_embeddings("../outputs/face_0.jpg", "cloak_face.jpg")
