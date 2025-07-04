import numpy as np
import torch
import cv2
from pathlib import Path
from main import MusicScoreDataset
from main import CRNN
import json
import random
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

def random_affine(img):
    translate = (random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05))  

    scale = random.uniform(0.95, 1.05)

    shear = random.uniform(-8, 8)

    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) 
    img_tensor = TF.to_pil_image(img_tensor)
    img_tensor = TF.affine(img_tensor, angle=0, translate=(int(translate[0]*img.shape[1]), int(translate[1]*img.shape[0])), scale=scale, shear=shear, fill=0)
    img_tensor = TF.to_tensor(img_tensor).squeeze(0) 
    return img_tensor.numpy()


def load_vocabulary_from_file(vocab_path):
    with open(vocab_path, 'r') as f:
        token_to_idx = json.load(f)
    vocab = sorted([token for token in token_to_idx if token != "<BLANK>"])
    idx_to_token = {int(idx): token for token, idx in token_to_idx.items()}
    return vocab, token_to_idx, idx_to_token

def ctc_decode(token_list):
    """
    Collapse consecutive duplicates and remove <BLANK> tokens for CTC decoding.
    """
    decoded = []
    prev_token = None
    for token in token_list:
        if token != prev_token and token != "<BLANK>":
            decoded.append(token)
        prev_token = token
    return decoded

# ===== CONFIGURATION =====
MODEL_PATH = "/Users/leosvjetlicic/Desktop/Diplomski/models/25.6.lr 0.05_ok_but_needs_more_epochs/crnn_epoch_10.pth"
IMAGE_PATH = "/Users/leosvjetlicic/Desktop/tst.png"
VOCAB_PATH = "/Users/leosvjetlicic/Desktop/Diplomski/vocab.json"
# =========================

# Load vocabulary
vocab = load_vocabulary_from_file(VOCAB_PATH)[0]
idx_to_char = {i+1: char for i, char in enumerate(vocab)}
idx_to_char[0] = "<BLANK>"

# Initialize model
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
model = CRNN(vocab_size=len(vocab)+1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Preprocess image
if not Path(IMAGE_PATH).exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]

# Resize with aspect ratio preservation
img = random_affine(img)

h, w = img.shape
new_h = 128
new_w = max(1, int(w * (new_h / h)))
img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Convert to tensor and add batch/channel dimensions
input_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]

with torch.no_grad():
    output = model(input_tensor)

output_indices = output.argmax(dim=2)[0].cpu().numpy()
predicted_chars = [idx_to_char[idx] for idx in output_indices]

# --- CTC decoding step ---
decoded_prediction = ctc_decode(predicted_chars)

print(f"Final prediction: {'\t'.join(decoded_prediction)}")

plt.imshow(img)
plt.show()