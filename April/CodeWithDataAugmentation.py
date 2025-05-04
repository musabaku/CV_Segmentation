import os
import glob
import torch
import nrrd
import nibabel as nib
import scipy.ndimage
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import re
import matplotlib.pyplot as plt
import copy  # For safely copying headers
import time  # For tracking training time
from contextlib import nullcontext  # For no-op context on CPU
import torchio as tio

# --- Configuration ---
INPUT_SHAPE = (128, 128, 64)   # Resize images to this shape
BATCH_SIZE = 2
EPOCHS = 1000                  # Number of training epochs (adjust as needed)
LEARNING_RATE = 1e-4
NUM_CLASSES = 4              # (class 0 = background, classes 1..3 = foreground)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False                # Set to True for debug prints/visualizations
# Check CUDA and cuDNN
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"cuDNN enabled before config: {torch.backends.cudnn.enabled}")
print(f"cuDNN version before config: {torch.backends.cudnn.version()}")

# --- cuDNN Optimizations ---
torch.backends.cudnn.enabled     = True
torch.backends.cudnn.benchmark   = True
torch.backends.cudnn.deterministic = False

# Confirm cuDNN is on and which version you’re using
if torch.backends.cudnn.enabled:
    print(f"✅ cuDNN is enabled (v{torch.backends.cudnn.version()}) and will be used for acceleration")
else:
    print("❌ cuDNN is NOT enabled; training will fall back to generic CUDA kernels")


if torch.cuda.is_available():
    print("✅ CUDA is available! Code will run on GPU.")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("❌ CUDA is not available. Code will run on CPU.")

# --- Directory Configuration ---
# Raw data directories (for val and test)
RAW_DATA_DIR = r"C:\Users\ISU\cv_dataset\AlternativeDatasetCropped"
VAL_DIR = os.path.join(RAW_DATA_DIR, "val")
TEST_DIR = os.path.join(RAW_DATA_DIR, "test")
# Preprocessed (cached) training data directory
CACHED_TRAIN_DIR = r"C:\Users\ISU\cv_dataset\AlternativeDatasetCropped\ProcessedData\Train"
# Output directory for predictions
OUTPUT_MASK_PATH = os.path.join(TEST_DIR, "predictionsCUDATest22ndApril")
os.makedirs(OUTPUT_MASK_PATH, exist_ok=True)

# --- Utility Functions for Raw Data ---
def load_medical_image(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext in ['nii', 'gz']:
        img = nib.load(file_path)
        data = img.get_fdata()
    elif ext == 'nrrd':
        data, _ = nrrd.read(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    return np.array(data)

def preprocess_image(img, shape=INPUT_SHAPE, order=1):
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    zoom_factors = [shape[i] / img.shape[i] for i in range(3)]
    return scipy.ndimage.zoom(img, zoom_factors, order=order)

def preprocess_mask(mask, shape=INPUT_SHAPE, order=0):
    zoom_factors = [shape[i] / mask.shape[i] for i in range(3)]
    return scipy.ndimage.zoom(mask, zoom_factors, order=order)

def get_image_mask_pairs(directory):
    image_files = glob.glob(os.path.join(directory, "images", "*"))
    mask_files = glob.glob(os.path.join(directory, "masks", "*"))
    
    def get_basename(file_path):
        base = os.path.basename(file_path)
        base = re.sub(r'\((SCAN|MASK|Mask)\)_cropped\.nrrd$', '', base, flags=re.IGNORECASE)
        return base

    image_dict = {get_basename(f): f for f in image_files}
    mask_dict = {get_basename(f): f for f in mask_files}
    
    common_keys = set(image_dict.keys()) & set(mask_dict.keys())
    images = [image_dict[key] for key in common_keys]
    masks = [mask_dict[key] for key in common_keys]
    if len(common_keys) == 0:
        print("❌ No matching images and masks found. Check filenames!")
    else:
        print(f"🔍 Found {len(images)} image-mask pairs in {directory}")
    return images, masks

# --- TorchIO Augmentations ---
train_transform = tio.Compose([
    # spatial
    tio.RandomFlip(axes=(0,1,2),    flip_probability=0.5),
    tio.RandomAffine(
        scales=(0.9,1.1),
        degrees=15,
        translation=5,
    ),
    tio.RandomElasticDeformation(
        num_control_points=7,
        max_displacement=7.5,
        locked_borders=2,
    ),
    # intensity
    tio.RandomNoise(std=(0,0.05)),
    tio.RandomBiasField(coefficients=(0.3,0.6)),
    tio.RandomGamma(log_gamma=(-0.3,0.3)),
    tio.ZNormalization(),  # zero-mean, unit-variance
])

# --- Dataset Classes ---

# For training, load preprocessed data (cached as .npy files)
# --- Dataset Classes ---
class TorchIODataset(Dataset):
    """Loads preprocessed .npy volumes and applies TorchIO transforms."""
    def __init__(self, processed_dir, transform=None):
        self.img_paths  = sorted(glob.glob(os.path.join(processed_dir, "*_img.npy")))
        self.mask_paths = [p.replace("_img.npy","_mask.npy") for p in self.img_paths]
        self.transform  = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img  = np.load(self.img_paths[idx]).astype(np.float32)
        mask = np.load(self.mask_paths[idx]).astype(np.int64)

        # normalize intensities to [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # add channel dim for TorchIO: shape (1,H,W,D)
        img_tensor  = torch.from_numpy(img)[None]
        mask_tensor = torch.from_numpy(mask)[None]

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img_tensor),
            mask =tio.LabelMap(  tensor=mask_tensor),
        )
        if self.transform:
            subject = self.transform(subject)

        return subject.image.data, subject.mask.data.squeeze(0).long()

# For validation and testing, process data on the fly
class MedicalDataset(Dataset):
    def __init__(self, directory):
        # directory should contain "images" and "masks" subfolders
        self.image_paths, self.mask_paths = get_image_mask_pairs(directory)
    
    def __len__(self):
        return len(self.image_paths)
    
    def resize_3d(self, img, shape, order=0):
        zoom_factors = [shape[i] / img.shape[i] for i in range(3)]
        return scipy.ndimage.zoom(img, zoom_factors, order=order)
    
    def __getitem__(self, idx):
        img = load_medical_image(self.image_paths[idx])
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = self.resize_3d(img, INPUT_SHAPE, order=1)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        mask = load_medical_image(self.mask_paths[idx])
        mask = self.resize_3d(mask, INPUT_SHAPE, order=0)
        mask = torch.tensor(mask, dtype=torch.long)
        return img, mask

# --- Model Definition (3D U-Net) ---
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.enc1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = conv_block(128, 256)
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(64, 32)
        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        dec3 = self.up3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        logits = self.out_conv(dec1)
        if DEBUG:
            print("=== Forward Pass Debug ===")
            print("Input shape:", x.shape)
            print("Enc1 shape:", enc1.shape)
            print("Enc2 shape:", enc2.shape)
            print("Enc3 shape:", enc3.shape)
            print("Bottleneck shape:", bottleneck.shape)
            print("Logits shape:", logits.shape)
            print("==========================")
        return logits

# --- Focal Tversky Loss ---
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, preds, targets):
        preds = F.softmax(preds, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
        dims = (2, 3, 4)
        TP = (preds * targets_one_hot).sum(dim=dims)
        FP = (preds * (1 - targets_one_hot)).sum(dim=dims)
        FN = ((1 - preds) * targets_one_hot).sum(dim=dims)
        Tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        focal_tversky_loss = (1 - Tversky) ** self.gamma
        return focal_tversky_loss.mean()

criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)

# --- Training and Evaluation ---
def compute_metrics(pred, target, num_classes=NUM_CLASSES, smooth=1e-6):
    dice_scores = []
    jaccard_scores = []
    per_class_metrics = {}
    for cls in range(num_classes):
        pred_cls = (pred == cls).astype(np.float32)
        target_cls = (target == cls).astype(np.float32)
        intersection = np.sum(pred_cls * target_cls)
        union = np.sum(pred_cls) + np.sum(target_cls)
        dice = (2 * intersection + smooth) / (union + smooth)
        jaccard = (intersection + smooth) / (np.sum(pred_cls) + np.sum(target_cls) - intersection + smooth)
        dice_scores.append(dice)
        jaccard_scores.append(jaccard)
        per_class_metrics[f'class_{cls}'] = {'dice': dice, 'jaccard': jaccard}
    pixel_accuracy = np.mean(pred.flatten() == target.flatten())
    mean_dice = np.mean(dice_scores)
    mean_jaccard = np.mean(jaccard_scores)
    return per_class_metrics, mean_dice, mean_jaccard, pixel_accuracy

def main():
    # Create datasets and loaders
    # Use cached training data
    train_dataset = TorchIODataset(CACHED_TRAIN_DIR, transform=train_transform)
    # Validation and test are processed on the fly from raw data
    val_dataset = MedicalDataset(VAL_DIR)
    test_dataset = MedicalDataset(TEST_DIR)
    
    # On Windows, num_workers=0 to avoid worker issues; on Linux, you can increase it.
    num_workers = 0 if os.name == 'nt' else 4
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    # Initialize model and optimizer
    model = UNet3D(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting model training on cached training data...")
    start_time = time.time()
    scaler = torch.amp.GradScaler(device=DEVICE)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            context = torch.amp.autocast("cuda") if DEVICE.type == "cuda" else nullcontext()
            with context:
                outputs = model(imgs)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_dataset)
        
        # Validation phase (on-the-fly processing)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                context = torch.amp.autocast("cuda") if DEVICE.type == "cuda" else nullcontext()
                with context:
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_dataset)
        
        print(f"Epoch {epoch+1}/{EPOCHS} -- Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    print("\n--- Training Completed ---")
    print(f"Total Training Time: {elapsed_time:.2f} seconds (≈ {hours}h {minutes}m {seconds:.2f}s)")
    
    # --- Inference & Evaluation on Test Set ---
    model.eval()
    dice_list = []
    iou_scores = []
    accuracies = []
    per_class_results = []
    
    # We process test data on the fly
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        with torch.no_grad():
            output = model(imgs)
            probabilities = F.softmax(output, dim=1)
            predicted_mask = torch.argmax(probabilities, dim=1)
        per_class_metrics, mean_dice, mean_jaccard, pixel_accuracy = compute_metrics(
            predicted_mask.cpu().numpy().squeeze(0),
            masks.cpu().numpy().squeeze(0)
        )
        dice_list.append(mean_dice)
        iou_scores.append(mean_jaccard)
        accuracies.append(pixel_accuracy)
        per_class_results.append(per_class_metrics)
    
    print("\n--- Test Evaluation Metrics ---")
    print(f"Mean Dice Coefficient: {np.mean(dice_list):.4f}")
    print(f"Mean Jaccard Index (IoU): {np.mean(iou_scores):.4f}")
    print(f"Mean Pixel Accuracy: {np.mean(accuracies):.4f}")

# Print per-class metrics
    print("Per-Class Metrics:")
    for class_id in range(NUM_CLASSES):
        dice_scores_per_class = [result[f'class_{class_id}']['dice'] for result in per_class_results]
        jaccard_scores_per_class = [result[f'class_{class_id}']['jaccard'] for result in per_class_results]
    
        mean_dice_per_class = np.mean(dice_scores_per_class)
        mean_jaccard_per_class = np.mean(jaccard_scores_per_class)
    
        print(f"Class {class_id}: Dice = {mean_dice_per_class:.4f}, Jaccard = {mean_jaccard_per_class:.4f}")

    # Optionally, save predicted masks in a format compatible with 3D Slicer.
    # Here, we process one test sample for demonstration.
    for img_path, mask_path in zip(get_image_mask_pairs(TEST_DIR)[0], get_image_mask_pairs(TEST_DIR)[1]):
        img = load_medical_image(img_path)
        mask_gt = load_medical_image(mask_path)
        img_proc = preprocess_image(img, INPUT_SHAPE, order=1)
        img_tensor = torch.tensor(img_proc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_mask = torch.argmax(probabilities, dim=1)
        # Resample prediction to original shape
        basename = os.path.basename(img_path)
        if basename.endswith('.nrrd'):
            output_filename = os.path.join(OUTPUT_MASK_PATH, basename.replace(".nrrd", "_pred_resampled.nrrd"))
        else:
            output_filename = os.path.join(OUTPUT_MASK_PATH, basename + "_pred_resampled.nrrd")
        original_data, original_header = nrrd.read(img_path)
        pred_mask_np = predicted_mask.cpu().numpy().squeeze(0)
        orig_shape = original_data.shape
        zoom_factors = [orig_shape[i] / pred_mask_np.shape[i] for i in range(3)]
        resampled_pred_mask = scipy.ndimage.zoom(pred_mask_np, zoom_factors, order=0)
        new_header = copy.deepcopy(original_header)
        new_header['type'] = 'short'
        if 'data file' in new_header:
            del new_header['data file']
        if 'encoding' in new_header:
            del new_header['encoding']
        nrrd.write(output_filename, resampled_pred_mask, header=new_header)
        print(f"3D Slicer-compatible mask saved to: {output_filename}")

if __name__ == "__main__":
    main()

