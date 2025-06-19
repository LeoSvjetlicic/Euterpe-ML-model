import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import random
import os
import json
from torch.utils.data import ConcatDataset, random_split
import torchvision.transforms.functional as TF
from Levenshtein import distance as levenshtein_distance
import matplotlib.pyplot as plt
# Add these imports at the top
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def calculate_metrics(predicted_tokens, true_tokens):
    """Calculate multiple evaluation metrics"""
    # Sequence Error Rate (SER)
    ser = 1.0 if predicted_tokens == true_tokens else 0.0

    # Symbol Error Rate (SyER) using Levenshtein distance
    syer = levenshtein_distance(''.join(predicted_tokens), 
                               ''.join(true_tokens)) / max(len(true_tokens), 1)

    # Token-level metrics (align sequences first)
    max_len = max(len(predicted_tokens), len(true_tokens))
    p = predicted_tokens + ['<PAD>']*(max_len - len(predicted_tokens))
    t = true_tokens + ['<PAD>']*(max_len - len(true_tokens))

    # Precision, Recall, F1
    precision = precision_score(t, p, average='micro', zero_division=0)
    recall = recall_score(t, p, average='micro', zero_division=0)
    f1 = f1_score(t, p, average='micro', zero_division=0)

    return {
        'ser': ser,
        'syer': syer,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_model(model, val_loader, idx_to_token, device):
    """Enhanced evaluation with multiple metrics"""
    model.eval()
    total_metrics = {
        'ser': 0.0,
        'syer': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0
    }
    total_samples = 0
    
    with torch.no_grad():
        for images, labels, label_lengths in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            for i in range(images.size(0)):
                # Decode predictions
                pred = ctc_decode(outputs[:, i:i+1, :], idx_to_token)
                target = [idx_to_token[idx.item()] 
                         for idx in labels[i] if idx.item() != 0]
                
                # Calculate metrics
                metrics = calculate_metrics(pred, target)
                for k in total_metrics:
                    total_metrics[k] += metrics[k]
                total_samples += 1

    # Average metrics
    avg_metrics = {k: v/total_samples for k, v in total_metrics.items()}
    
    # Write to file
    with open("evaluation.txt", "w") as f:
        f.write("Evaluation Metrics:\n")
        f.write(f"- Sequence Error Rate: {avg_metrics['ser']*100:.2f}%\n")
        f.write(f"- Symbol Error Rate: {avg_metrics['syer']*100:.2f}%\n")
        f.write(f"- Token Precision: {avg_metrics['precision']*100:.2f}%\n")
        f.write(f"- Token Recall: {avg_metrics['recall']*100:.2f}%\n")
        f.write(f"- Token F1 Score: {avg_metrics['f1']*100:.2f}%\n")
    
    return avg_metrics

def load_vocabulary_from_file(vocab_path):
    with open(vocab_path, 'r') as f:
        token_to_idx = json.load(f)
    vocab = sorted([token for token in token_to_idx if token != "<BLANK>"])
    idx_to_token = {int(idx): token for token, idx in token_to_idx.items()}
    return vocab, token_to_idx, idx_to_token

def calculate_cer(predicted, target):
    """Character Error Rate (CER) using Levenshtein distance"""
    return levenshtein_distance(''.join(predicted), ''.join(target)) / max(len(target), 1)

def calculate_metrics(predicted_tokens, true_tokens):
    cer = calculate_cer(predicted_tokens, true_tokens)
    sequence_accuracy = 1.0 if predicted_tokens == true_tokens else 0.0
    return cer, sequence_accuracy

# def evaluate_model(model, val_loader, idx_to_token, device):
#     model.eval()
#     total_cer, total_seq_acc, total = 0.0, 0.0, 0
#     with torch.no_grad():
#         for images, labels, label_lengths in val_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             for i in range(images.size(0)):
#                 pred = ctc_decode(outputs[:, i:i+1, :], idx_to_token)  # Keep batch dimension
#                 target = [idx_to_token[idx.item()] for idx in labels[i] if idx.item() != 0]
#                 cer, seq_acc = calculate_metrics(pred, target)
#                 total_cer += cer
#                 total_seq_acc += seq_acc
#                 total += 1
#     avg_cer = total_cer / total
#     avg_seq_acc = total_seq_acc / total
#     # Write to file
#     with open("evaluation.txt", "w") as f:
#         f.write(f"Validation CER: {avg_cer:.4f}\n")
#         f.write(f"Validation Sequence Accuracy: {avg_seq_acc:.4f}\n")


def random_affine(img):
    # Random rotation between -30 and 30 degrees (sometimes full 180 for upside down)
    if random.random() < 0.01:  # 10% chance for upside down
        angle = random.choice([180, 0])
    else:
        angle = random.uniform(-15, 15)

    # Random translation (shift)
    translate = (random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05))  # up to 5% shift

    # Random scale (zoom in/out)
    scale = random.uniform(0.95, 1.05)

    # Random shear (diagonal distortion)
    shear = random.uniform(-8, 8)

    # Convert numpy to torch tensor and back for torchvision
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
    img_tensor = TF.to_pil_image(img_tensor)
    img_tensor = TF.affine(img_tensor, angle=angle, translate=(int(translate[0]*img.shape[1]), int(translate[1]*img.shape[0])), scale=scale, shear=shear, fill=0)
    img_tensor = TF.to_tensor(img_tensor).squeeze(0)  # [H, W]
    return img_tensor.numpy()

# Custom Dataset
class MusicScoreDataset(Dataset):
    def __init__(self, image_dir, transform=None, vocab=None, max_seq_len=65, num_samples=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.max_seq_len = max_seq_len
        all_image_paths = sorted([p for p in self.image_dir.rglob("*.png") if p.with_suffix('.semantic').exists()])
        if num_samples is not None:
            self.image_paths = random.sample(all_image_paths, num_samples)
        else:
            self.image_paths = all_image_paths
        self.label_paths = [p.with_suffix('.semantic') for p in self.image_paths]
        if vocab is None:
            self.vocab = self.build_vocab()
            self.token_to_idx = {token: idx + 1 for idx, token in enumerate(self.vocab)}
            self.token_to_idx["<BLANK>"] = 0
            # Save vocabulary if vocab_save_path is provided
            os.makedirs(os.path.dirname("/Users/leosvjetlicic/Desktop/Diplomski/vocab"), exist_ok=True)
            with open("vocab_save_path", 'w') as f:
                json.dump(self.token_to_idx, f, indent=4)
        else:
            self.vocab = vocab
            self.token_to_idx = {token: idx + 1 for idx, token in enumerate(self.vocab)}
            self.token_to_idx["<BLANK>"] = 0
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
    def build_vocab(self):
        """Build vocabulary from all .semantic files in the dataset."""
        vocab = set()
        for label_path in self.label_paths:
            with open(label_path, "r") as f:
                tokens = f.read().strip().split()
                vocab.update(tokens)
        return sorted(list(vocab))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get an image-label pair."""
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image not found: {img_path}")
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img = img / 255.0  # Normalize to [0, 1]

        # Resize image while maintaining aspect ratio
        original_height, original_width = img.shape
        if original_height == 0 or original_width == 0:
            raise ValueError(f"Invalid image dimensions: {original_height}x{original_width} at {img_path}")
        aspect_ratio = original_width / original_height
        new_height = 128
        new_width = max(1, int(aspect_ratio * new_height))  # Ensure width is at least 1
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            # In your __getitem__ method, after resizing and before converting to torch:
        if self.transform:
            img = self.transform(img)
        else:
            # Apply random distortion with some probability
            if random.random() < 0.5:
                img = random_affine(img)

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W)
        if self.transform:
            img = self.transform(img)

        label_path = self.label_paths[idx]
        with open(label_path, "r") as f:
            tokens = f.read().strip().split()
        label = [self.token_to_idx[token] for token in tokens if token in self.token_to_idx]
        label = label + [0] * (self.max_seq_len - len(label))  # Pad with blank
        label = torch.tensor(label[:self.max_seq_len], dtype=torch.int32)

        return img, label, len(tokens)

class CRNN(nn.Module):
    def __init__(self, vocab_size):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),  # output: (32, 64, W/2)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),  # output: (64, 32, W/4)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),  # output: (128, 16, W/8)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),  # output: (256, 8, W/16)
        )

        # After conv layers, height is reduced from 128 to 8
        # So input to RNN is (W/16, 256*8) per timestep
        self.rnn_input_size = 256 * 8
        self.rnn = nn.LSTM(input_size=self.rnn_input_size, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(256 * 2, vocab_size)  # bidirectional, so 2x hidden size

    def forward(self, x):
        x = self.cnn(x)  # Shape: (B, C, H=8, W/16)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x = x.contiguous().view(b, w, c * h)  # (B, W, C*H) â†’ sequence length W
        x, _ = self.rnn(x)  # output shape: (B, W, 512)
        x = self.fc(x)      # output shape: (B, W, vocab_size)
        return x


def collate_fn(batch):
    """
    Pad images to the maximum width in the batch.

    Args:
        batch: List of (image, label, label_length) tuples.

    Returns:
        Tuple of padded images, labels, and label lengths.
    """
    images, labels, label_lengths = zip(*batch)
    max_height = max(img.size(1) for img in images)
    max_width = max(img.size(2) for img in images)
    padded_images = []
    for img in images:
        pad_height = max_height - img.size(1)
        pad_width = max_width - img.size(2)
        padded_img = F.pad(img, (0, pad_width, 0, pad_height), mode='constant', value=0)
        padded_images.append(padded_img)
    padded_images = torch.stack(padded_images)
    labels = torch.stack(labels)
    label_lengths = torch.tensor(label_lengths)
    return padded_images, labels, label_lengths

# Training Function
def train_model(model, train_loader, val_loader, num_epochs=12, device="cuda"):
    """Train the CRNN model and plot loss curves."""
    model = model.to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adadelta(model.parameters(),lr=0.3)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels, label_lengths in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.log_softmax(2)
            input_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.long).to(device)
            loss = criterion(outputs.permute(1, 0, 2), labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels, label_lengths in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = outputs.log_softmax(2)
                input_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.long).to(device)
                loss = criterion(outputs.permute(1, 0, 2), labels, input_lengths, label_lengths)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        checkpoint_dir = "/Users/leosvjetlicic/Desktop/Diplomski/models/"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{checkpoint_dir}/crnn_epoch_{epoch+1}.pth")

    # === Plotting after training ===
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png", dpi=200)
    plt.close()


def preprocess_image(image_path, height=128):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = img / 255.0
    original_height, original_width = img.shape
    if original_height == 0 or original_width == 0:
        raise ValueError(f"Invalid image dimensions: {original_height}x{original_width}")
    aspect_ratio = original_width / original_height
    new_width = max(1, int(aspect_ratio * height))
    img = cv2.resize(img, (new_width, height), interpolation=cv2.INTER_AREA)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return img

# Enhanced CTC decoding with debugging
def ctc_decode(logits, idx_to_token):
    """CTC greedy decoder that handles 2D and 3D inputs"""
    # Ensure 3D shape: [seq_len, batch=1, vocab_size]
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)  # Add batch dimension
    
    # Permute to [batch, seq_len, vocab_size]
    logits = logits.permute(1, 0, 2)
    
    # Greedy decoding
    output_indices = logits.argmax(dim=2)[0].cpu().numpy()
    decoded = []
    prev_idx = None
    for idx in output_indices:
        if idx != prev_idx and idx != 0:  # 0 is <BLANK>
            decoded.append(idx_to_token[idx])
        prev_idx = idx
    return decoded

# Inference function
def predict_music_score(model, image_path, idx_to_token, device="cpu"):
    model.eval()
    img = preprocess_image(image_path).to(device)
    with torch.no_grad():
        logits = model(img)  # (batch=1, seq_len, vocab_size)
        logits = logits.log_softmax(2)
        decoded = ctc_decode(logits, idx_to_token)
    return decoded

# Load vocabulary from file
def load_vocabulary(vocab_path):
    with open(vocab_path, 'r') as f:
        token_to_idx = json.load(f)
    vocab = sorted([token for token, idx in token_to_idx.items() if token != "<BLANK>"])
    idx_to_token = {int(idx): token for token, idx in token_to_idx.items()}
    return vocab, idx_to_token

# Main Script
if __name__ == "__main__":
    # Dataset paths (adjust to your PrIMuS dataset structure)
    data_package_aa = "/Users/leosvjetlicic/Desktop/Diplomski/primusCalvoRizoAppliedSciences2018/package_aa"  # Contains subdirs with .png and .semantic
    data_package_ab = "/Users/leosvjetlicic/Desktop/Diplomski/primusCalvoRizoAppliedSciences2018/package_ab"    # Contains subdirs with .png and .semantic

    # Load datasets
    # Combine datasets and split
    train_dataset = MusicScoreDataset(data_package_aa, transform=None, num_samples=None)
    val_dataset = MusicScoreDataset(data_package_ab, transform=None, vocab=train_dataset.vocab, num_samples=None)
    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    total_samples = len(combined_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    train_split, val_split = random_split(
        combined_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Create new DataLoaders
    train_loader = DataLoader(
        train_split,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_split,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )
    # Print dataset info
    print(f"Training samples: {len(train_split)}")
    print(f"Validation samples: {len(val_split)}")
    print(f"Vocabulary size: {len(train_dataset.vocab)}")

    # Initialize model
    model = CRNN(vocab_size=len(train_dataset.vocab) + 1)  # +1 for blank token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=10, device=device)

    VOCAB_PATH = "/Users/leosvjetlicic/Desktop/Diplomski/vocab_save_path"
    
    idx_to_token = load_vocabulary_from_file(VOCAB_PATH)[2]
    evaluate_model(model, val_loader, idx_to_token, device)
