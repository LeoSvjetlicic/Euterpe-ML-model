import numpy as np
import torch
import cv2
from pathlib import Path
from main import MusicScoreDataset
from main import CRNN

# ===== CONFIGURATION =====
MODEL_PATH = "/Users/leosvjetlicic/Desktop/Diplomski/models/withEvaluation_14-6/crnn_epoch_3.pth"
TRAIN_DATA = "/Users/leosvjetlicic/Desktop/Diplomski/primusCalvoRizoAppliedSciences2018/package_aa"
IMAGE_PATH = "/Users/leosvjetlicic/Desktop/Diplomski/primusCalvoRizoAppliedSciences2018/package_aa/000135389-1_1_1/000135389-1_1_1.png"
# =========================

# Load vocabulary (adjust based on your actual vocab format)
# Example: Rebuild vocab from your dataset

dataset = MusicScoreDataset(TRAIN_DATA, transform=None, num_samples=None)
vocab = dataset.vocab  # This should be a list of unique characters/tokens

idx_to_char = {i+1: char for i, char in enumerate(vocab)}
idx_to_char[0] = "<BLANK>"

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(vocab_size=len(vocab)+1).to(device)  # Assuming your CRNN class is defined
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Preprocess image
if not Path(IMAGE_PATH).exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# Image processing pipeline (matches your dataset preprocessing)
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]

# Resize with aspect ratio preservation
h, w = img.shape
new_h = 128
new_w = max(1, int(w * (new_h / h)))
img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Convert to tensor and add batch/channel dimensions
input_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]

# Run inference
with torch.no_grad():
    output = model(input_tensor)
    
# Print raw output tensor
print("\n=== Raw Model Output Tensor ===")
print(f"Shape: {output.shape}")
print(output)

# Decode predictions
output_indices = output.argmax(dim=2)[0].cpu().numpy()
predicted_chars = [idx_to_char[idx] for idx in output_indices]

print(f"Final prediction: {'\t'.join([c for c in predicted_chars if c != '<BLANK>'])}")
