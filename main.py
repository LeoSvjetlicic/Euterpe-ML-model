import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split, Subset
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import random
import json
from torchvision.utils import save_image
import os
from PIL import Image
import io
import torchvision.transforms.functional as TF

def load_vocabulary_from_file(vocab_path):
    with open(vocab_path, 'r') as f:
        token_to_idx = json.load(f)
    vocab = sorted([token for token in token_to_idx if token != "<BLANK>"])
    idx_to_token = {int(idx): token for token, idx in token_to_idx.items()}
    return vocab, token_to_idx, idx_to_token

def random_camera_augment(img):
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    img_pil = TF.to_pil_image(img_tensor)
    
    # Random affine transformation
    angle = random.uniform(-0.8, 0.8)
    img_height, img_width = img.shape
    max_translate_x = min(8, int(img_width * 0.10))
    max_translate_y = min(8, int(img_height * 0.10))
    translate_x = random.uniform(-max_translate_x, max_translate_x)
    translate_y = random.uniform(-max_translate_y, max_translate_y)
    scale = random.uniform(0.95, 1.10)
    shear = random.uniform(-4.5, 4.5)
    
    img_pil = TF.affine(
        img_pil,
        angle=angle,
        translate=(int(translate_x), int(translate_y)),
        scale=scale,
        shear=shear,
        fill=0
    )
    
    # Brightness and contrast adjustment
    if random.random() < 0.5:
        brightness = random.uniform(0.7, 1.3)
        img_pil = TF.adjust_brightness(img_pil, brightness)
        contrast = random.uniform(0.7, 1.3)
        img_pil = TF.adjust_contrast(img_pil, contrast)
    
    img_tensor = TF.to_tensor(img_pil).squeeze(0)
    img_array = img_tensor.numpy()
    
    
    # No augmentations: tryIndex = 0
    # Only blur: tryIndex = 1
    # Only noise: tryIndex = 2
    # Only occlusion: tryIndex = 4
    # Blur + noise: tryIndex = 3
    # Blur + occlusion: tryIndex = 5
    # Noise + occlusion: tryIndex = 6
    # All three: tryIndex = 7

    # Gaussian blur
    if random.random() < 0.3:
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.5, 2.0)
        img_uint8 = (img_array * 255).astype(np.uint8)
        img_blurred = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), sigma)
        img_array = img_blurred.astype(np.float32) / 255.0
    
    # Add Gaussian noise
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.02, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 1)
    
    # Random occlusion
    if random.random() < 0.3:
        h, w = img_array.shape
        occ_w = random.randint(w//10, w//5)
        occ_h = random.randint(h//10, h//5)
        x0 = random.randint(0, w - occ_w)
        y0 = random.randint(0, h - occ_h)
        img_array[y0:y0 + occ_h, x0:x0 + occ_w] = random.uniform(0, 0.3)
    
    return img_array

class MusicScoreDataset(Dataset):
    def __init__(self, image_dir, transform=None, vocab=None, max_seq_len=65, num_samples=None,
                 augment_affine=False, augment_noise=False):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.augment_affine = augment_affine
        self.augment_noise = augment_noise
        all_image_paths = sorted([p for p in self.image_dir.rglob("*.png") if p.with_suffix('.semantic').exists()])
        self.image_paths = random.sample(all_image_paths, num_samples) if num_samples else all_image_paths
        self.label_paths = [p.with_suffix('.semantic') for p in self.image_paths]

        if vocab is None:
            self.vocab = self.build_vocab()
            self.token_to_idx = {token: idx + 1 for idx, token in enumerate(self.vocab)}
            self.token_to_idx["<BLANK>"] = 0
            os.makedirs("/Users/leosvjetlicic/Desktop/Diplomski", exist_ok=True)
            with open("/Users/leosvjetlicic/Desktop/Diplomski/vocab.json", 'w') as f:
                json.dump(self.token_to_idx, f, indent=4)
        else:
            self.vocab = vocab
            self.token_to_idx = {token: idx + 1 for idx, token in enumerate(self.vocab)}
            self.token_to_idx["<BLANK>"] = 0

        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

    def build_vocab(self):
        vocab = set()
        for label_path in self.label_paths:
            with open(label_path, "r") as f:
                tokens = f.read().strip().split()
                vocab.update(tokens)
        return sorted(list(vocab))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image not found: {img_path}")
        
        
        if random.random() < 0.8:
           img = random_camera_augment(img)
        
        original_height, original_width = img.shape
        if original_height == 0 or original_width == 0:
            raise ValueError(f"Invalid image dimensions at {img_path}")

        aspect_ratio = original_width / original_height
        new_height = 128
        new_width = max(1, int(aspect_ratio * new_height))
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            img = self.transform(img)

        label_path = self.label_paths[idx]
        with open(label_path, "r") as f:
            tokens = f.read().strip().split()
        label = [self.token_to_idx[token] for token in tokens if token in self.token_to_idx]
        label = torch.tensor(label, dtype=torch.int32)
        return img, label, len(label)


class CRNN(nn.Module):
    def __init__(self, vocab_size):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2), 

            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2), 

            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2), 

            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2), 
        )

        #  LSTM aktivacijska funkcija - softmax?????  nije preporuceno jer CTC u koji se predaju ove vrijednosti koristi oblik softmaxa pa je preporucene predati "RAW" vrijednosti
        # Ovo mi nije jasnoooo
        self.rnn_input_size = 256 * 6
        self.rnn = nn.LSTM(input_size=self.rnn_input_size, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(256 * 2, vocab_size)

    def forward(self, x):
        x = self.cnn(x)  # Shape: (B, C, H=8, W/16)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x = x.contiguous().view(b, w, c * h)  # (B, W, C*H) â†’ sequence length W
        x, _ = self.rnn(x)  # output shape: (B, W, 512)
        x = self.fc(x)      # output shape: (B, W, vocab_size)
        return x

#  normalizirati izlaze
def collate_fn(batch):
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
    max_label_len = max(len(label) for label in labels)
    padded_labels = torch.zeros(len(labels), max_label_len, dtype=torch.int32)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label
    label_lengths = torch.tensor(label_lengths)
    return padded_images, padded_labels, label_lengths

def train_model(model, train_loader, num_epochs=30, device="cpu"):
    model = model.to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adadelta(model.parameters(),lr=1)
    train_losses = []

    scaler = torch.amp.GradScaler(device="cuda")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels, label_lengths in train_loader:
            images, labels, label_lengths = images.to(device), labels.to(device), label_lengths.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images) 
                outputs = outputs.log_softmax(2)
                input_lengths = torch.full((images.size(0)), outputs.size(1), dtype=torch.long).to(device)
                loss = criterion(outputs.permute(1, 0, 2), labels, input_lengths, label_lengths)
                loss.backward()
            # backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
       
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, ")
        checkpoint_dir = "(../models/"
        os.makedirs(checkpoint_dir, exist_ok=True)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"{checkpoint_dir}/crnn_epoch_{epoch+1}.pth")

def decode_sequence(tensor_seq, idx_to_char):
    chars = [idx_to_char[idx.item()] for idx in tensor_seq]
    return ''.join(chars)

def ctc_decode_idx(token_list, blank_token=0):
    decoded = []
    prev_token = None
    for token in token_list:
        if token != prev_token and token != blank_token:
            decoded.append(token)
        prev_token = token
    return decoded

def evaluate_models(model_class, test_split, model_folder, device, vocab, batch_size=16):
    output_file = os.path.join(model_folder, "description.txt")
    
    with open(output_file, "w") as fdesc:
        def log(msg):
            print(msg)
            fdesc.write(msg + "\n")

        test_loader = DataLoader(
            test_split,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        token_accuracies = []
        sequence_accuracies = []
        
        model_files = [
            f for f in os.listdir(model_folder) if f.endswith('.pth')
        ]
        
        if not model_files:
            log("No model files found in the specified folder.")
            return {"avg_token_accuracy": 0.0, "avg_sequence_accuracy": 0.0}
        
        for model_file in model_files:
            log(f"Evaluating model: {model_file}")
            
            model = model_class(vocab_size=len(vocab) + 1)
            model.load_state_dict(torch.load(os.path.join(model_folder, model_file), map_location=device))
            model.to(device)
            model.eval()
            
            total_tokens = 0
            correct_tokens = 0
            total_sequences = 0
            correct_sequences = 0
            
            with torch.no_grad():
                for images, targets, lengths in test_loader:
                    images = images.to(device)
                    targets = targets.to(device)

                    output = model(images)  # [B, T, C]
                    predicted = output.argmax(dim=2)[0].cpu().numpy()

                    for i in range(images.size(0)):
                        seq_len = lengths[i]
                        target_seq = targets[i, :seq_len].cpu().tolist()

                        decoded_pred_seq = ctc_decode_idx(predicted)

                        min_len = min(len(decoded_pred_seq), len(target_seq))
                        correct_tokens += sum(p == t for p, t in zip(decoded_pred_seq[:min_len], target_seq[:min_len]))
                        total_tokens += len(target_seq)

                        if decoded_pred_seq == target_seq:
                            correct_sequences += 1
                        total_sequences += 1
            
            token_accuracy = (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0.0
            sequence_accuracy = (correct_sequences / total_sequences) * 100 if total_sequences > 0 else 0.0
            
            log(f"Model {model_file} - Token Accuracy: {token_accuracy:.2f}%, Sequence Accuracy: {sequence_accuracy:.2f}%")
            token_accuracies.append(token_accuracy)
            sequence_accuracies.append(sequence_accuracy)
        
        avg_token_accuracy = np.mean(token_accuracies) if token_accuracies else 0.0
        avg_sequence_accuracy = np.mean(sequence_accuracies) if sequence_accuracies else 0.0
        
        log("\nSummary of Evaluation:")
        log(f"Average Token Accuracy across {len(model_files)} models: {avg_token_accuracy:.2f}%")
        log(f"Average Sequence Accuracy across {len(model_files)} models: {avg_sequence_accuracy:.2f}%")
    
    return {
        "avg_token_accuracy": avg_token_accuracy,
        "avg_sequence_accuracy": avg_sequence_accuracy
    }

def save_normalized_images(dataset, output_path, num_images=50):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Pick a fixed subset (first N samples)
    indices = list(range(min(num_images, len(dataset))))
    subset = Subset(dataset, indices)

    for i, (img, _, _) in enumerate(subset):
        # Ensure image is a 3D tensor: (1, H, W)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3 and img.shape[0] != 1:
            raise ValueError("Expected a single-channel image.")

        # Save image
        file_path = output_path / f"image_{i:03d}.png"
        save_image(img, str(file_path))

    print(f"Saved {len(subset)} normalized images to: {output_path}")


if __name__ == "__main__":
    data_package_aa = "/Users/leosvjetlicic/Desktop/Diplomski/primusCalvoRizoAppliedSciences2018/package_aa" 
    data_package_ab = "/Users/leosvjetlicic/Desktop/Diplomski/primusCalvoRizoAppliedSciences2018/package_ab" 
    data_package_c = "/Users/leosvjetlicic/Desktop/Diplomski/Corpus" 

    VOCAB_PATH = "/Users/leosvjetlicic/Desktop/Diplomski/vocab.json" 
    vocab = load_vocabulary_from_file(VOCAB_PATH)[0]

    a_dataset = MusicScoreDataset(data_package_aa, transform=None, vocab=vocab, num_samples=None)
    b_dataset = MusicScoreDataset(data_package_ab, transform=None, vocab=vocab, num_samples=None)
    c_dataset = MusicScoreDataset(data_package_c, transform=None, vocab=vocab, num_samples=None)

    combined_dataset = ConcatDataset([a_dataset, b_dataset, c_dataset])
    total_samples = len(combined_dataset)
    test_size = int(0.05 * total_samples)
    train_and_validation_size = total_samples - test_size
    train_size = int(1 * train_and_validation_size)
    val_size = train_and_validation_size - train_size

    test_split, others_split = random_split(
        combined_dataset,
        [test_size, train_and_validation_size],
        generator=torch.Generator().manual_seed(42) 
    )

    train_split, val_split = random_split(
        others_split,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) 
    )

    train_loader = DataLoader(
        train_split,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Test samples: {len(test_split)}")
    print(f"Training samples: {len(train_split)}")
    print(f"Validation samples: {len(val_split)}")
    print(f"Vocabulary size: {len(vocab)}")

    device = torch.device("cuda")
    print(f"Using device: {device}")

    model = CRNN(vocab_size=len(vocab) + 1)
    train_model(model, train_loader, num_epochs=70, device=device)

    model_folder = "/Users/leosvjetlicic/Desktop/Diplomski/models"

    results = evaluate_models(
        model_class=CRNN,
        test_split=test_split,
        model_folder=model_folder,
        device=device,
        vocab = vocab,
        batch_size=16,
    )

