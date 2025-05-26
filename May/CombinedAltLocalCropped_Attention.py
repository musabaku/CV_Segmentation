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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import re
import matplotlib.pyplot as plt
import copy  # For safely copying headers
import time  # For tracking training time
from contextlib import nullcontext  # For no-op context on CPU
import torchio as tio
import sys
import random # For reproducibility

print("[NOTEBOOK_LOG] All Python packages imported successfully.")

# --- Configuration ---
print("[NOTEBOOK_LOG] Setting up configurations (INPUT_SHAPE, BATCH_SIZE, etc.)...")
INPUT_SHAPE = (128, 128, 64)  # Resize images to this shape (H, W, D)
BATCH_SIZE = 2
EPOCHS = 1000  # Number of training epochs
LEARNING_RATE = 1e-4
NUM_CLASSES = 4  # (class 0 = background, classes 1..3 = foreground)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False  # Set to True for debug prints/visualizations/anomaly detection
VERSION_SUFFIX = "SingleSource_NoHistStd_Robust_v4_FullScript" # Updated version for this run
SEED = 42 # For reproducibility
WEIGHT_DECAY = 1e-5 # For optimizer
GRAD_CLIP_MAX_NORM = 1.0 # For gradient clipping
SCHEDULER_PATIENCE = 15 # Patience for ReduceLROnPlateau
SCHEDULER_FACTOR = 0.2 # Factor for ReduceLROnPlateau
MIN_LR = 1e-7 # Minimum learning rate for scheduler

print("[NOTEBOOK_LOG] Basic configurations set.")

# --- Reproducibility ---
print(f"[NOTEBOOK_LOG] Setting random seeds to {SEED} for reproducibility...")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # The following two lines can make things slower if not strictly needed
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print("✅ CUDA is available! Code will run on GPU.")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("❌ CUDA is not available. Code will run on CPU.")
print(f"[NOTEBOOK_LOG] Device set to: {DEVICE}")


# --- Directory Configuration ---
print("[NOTEBOOK_LOG] Starting directory configuration (corrected paths)...")
RAW_DATA_PARENT_DIR = "/home/isu.femilda/hpc_project/CombinedAltLocalCropped" # MODIFY AS NEEDED

RAW_TRAIN_DIR = os.path.join(RAW_DATA_PARENT_DIR, "train")
RAW_VAL_DIR = os.path.join(RAW_DATA_PARENT_DIR, "val")
RAW_TEST_DIR = os.path.join(RAW_DATA_PARENT_DIR, "test")

PROCESSED_DATA_PARENT_DIR = os.path.join(RAW_DATA_PARENT_DIR, f"ProcessedData_{VERSION_SUFFIX}")
CACHED_TRAIN_DIR = os.path.join(PROCESSED_DATA_PARENT_DIR, "Train_cached")
CACHED_VAL_DIR = os.path.join(PROCESSED_DATA_PARENT_DIR, "Val_cached")
CACHED_TEST_DIR = os.path.join(PROCESSED_DATA_PARENT_DIR, "Test_cached")

OUTPUT_MASK_PATH = os.path.join(RAW_DATA_PARENT_DIR, f"predictions_AttentionUNet_{VERSION_SUFFIX}") # Changed to be relative to PARENT_DIR for common output location


print(f"[NOTEBOOK_LOG] RAW_DATA_PARENT_DIR set to: {RAW_DATA_PARENT_DIR}")
print(f"[NOTEBOOK_LOG] PROCESSED_DATA_PARENT_DIR (where cached data goes) set to: {PROCESSED_DATA_PARENT_DIR}")
print(f"[NOTEBOOK_LOG] OUTPUT_MASK_PATH (where predictions and models go) set to: {OUTPUT_MASK_PATH}")

os.makedirs(OUTPUT_MASK_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PARENT_DIR, exist_ok=True)
os.makedirs(CACHED_TRAIN_DIR, exist_ok=True)
os.makedirs(CACHED_VAL_DIR, exist_ok=True)
os.makedirs(CACHED_TEST_DIR, exist_ok=True)
print("[NOTEBOOK_LOG] Ensured all PROCESSED, CACHED, and OUTPUT data parent directories exist.")
print("[NOTEBOOK_LOG] Directory configuration complete.")


# --- Global Utility Functions ---
def get_basename(file_path):
    """Extracts a common basename from image/mask filenames."""
    base = os.path.basename(file_path)
    name_stem = base
    if name_stem.lower().endswith(".nii.gz"):
        name_stem = name_stem[:-len(".nii.gz")]
    elif name_stem.lower().endswith(".nrrd"):
        name_stem = name_stem[:-len(".nrrd")]
    elif name_stem.lower().endswith(".nii"):
        name_stem = name_stem[:-len(".nii")]

    suffixes_to_strip = [
        "_cropped",
        "(SCAN)", "(scan)",
        "(MASK)", "(mask)",
        "_SCAN", "_scan",
        "_MASK", "_mask",
        "SCAN", "scan",
        "MASK", "mask"
    ]

    modified_stem = name_stem
    for suffix in suffixes_to_strip:
        if modified_stem.lower().endswith(suffix.lower()):
            modified_stem = modified_stem[:-len(suffix)]
    return modified_stem.strip("_-")


# --- Utility Functions for Raw Data (used during initial caching) ---
def load_medical_image(file_path):
    """Loads a medical image from .nii, .nii.gz, or .nrrd formats."""
    ext_parts = file_path.split('.')
    ext = ext_parts[-1].lower()
    if len(ext_parts) > 2 and ext_parts[-2].lower() == 'nii' and ext == 'gz': # handle .nii.gz
        ext = 'nii.gz'

    if ext in ['nii', 'nii.gz']:
        img = nib.load(file_path)
        data = img.get_fdata()
    elif ext == 'nrrd':
        data, _ = nrrd.read(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path} (parsed ext: {ext})")
    return np.array(data)

def preprocess_image_raw(img, shape=INPUT_SHAPE, order=1):
    """Normalizes (instance-wise to [0,1]) and resizes an image."""
    img_min = img.min()
    img_max = img.max()
    # Instance-wise normalization to [0, 1]
    img = (img - img_min) / (img_max - img_min + 1e-8)
    zoom_factors = [shape[i] / img.shape[i] for i in range(len(shape))]
    return scipy.ndimage.zoom(img, zoom_factors, order=order, mode='nearest')

def preprocess_mask_raw(mask, shape=INPUT_SHAPE, order=0):
    """Resizes a mask."""
    zoom_factors = [shape[i] / mask.shape[i] for i in range(len(shape))]
    return scipy.ndimage.zoom(mask, zoom_factors, order=order, mode='nearest')

def get_image_mask_pairs(directory):
    """Finds matching image and mask files in a directory using the global get_basename."""
    image_files_path = os.path.join(directory, "images", "*")
    mask_files_path = os.path.join(directory, "masks", "*")

    image_files = sorted(glob.glob(image_files_path))
    mask_files = sorted(glob.glob(mask_files_path))

    if not image_files:
        print(f"⚠️ [NOTEBOOK_LOG - get_image_mask_pairs] No image files found in {os.path.join(directory, 'images')}")
    if not mask_files:
        print(f"⚠️ [NOTEBOOK_LOG - get_image_mask_pairs] No mask files found in {os.path.join(directory, 'masks')}")

    image_dict = {get_basename(f): f for f in image_files}
    mask_dict = {get_basename(f): f for f in mask_files}

    common_keys = sorted(list(set(image_dict.keys()) & set(mask_dict.keys())))

    paired_files = []
    for key in common_keys:
        paired_files.append((image_dict[key], mask_dict[key]))

    if not paired_files:
        print(f"❌ No matching images and masks found in {directory}. Check filenames, patterns, and if 'images'/'masks' subdirectories exist and contain files.")
    else:
        print(f"🔍 Found {len(paired_files)} image-mask pairs in {directory}")
    return paired_files

# --- TorchIO Augmentations (for Training) ---
print("[NOTEBOOK_LOG] Defining TorchIO augmentations...")
train_transform = tio.Compose([
    tio.RandomFlip(axes=(0,1,2), flip_probability=0.5),
    tio.RandomAffine(
        scales=(0.9,1.1),
        degrees=15,
        translation=5,
        default_pad_value=0.0,
        image_interpolation='linear',
    ),
    tio.RandomElasticDeformation(
        num_control_points=7,
        max_displacement=7.5,
        locked_borders=2,
    ),
    tio.RandomNoise(std=(0,0.05)),
    tio.RandomBiasField(coefficients=(0.3,0.6)),
    tio.RandomGamma(log_gamma=(-0.3,0.3)),
    tio.Clamp(out_min=0.0, out_max=1.0) # Ensure image data remains in [0,1] after augmentations
])
print("[NOTEBOOK_LOG] TorchIO augmentations defined.")

# --- Dataset Classes ---
print("[NOTEBOOK_LOG] Defining Dataset classes (TorchIODataset, PreprocessedDataset)...")
class TorchIODataset(Dataset):
    """Loads preprocessed .npy volumes from CACHED_TRAIN_DIR and applies TorchIO transforms."""
    def __init__(self, processed_dir, transform=None):
        self.img_paths  = sorted(glob.glob(os.path.join(processed_dir, "*_img.npy")))
        self.mask_paths = [p.replace("_img.npy","_mask.npy") for p in self.img_paths]
        self.transform  = transform
        if not self.img_paths:
            print(f"Warning: No .npy image files found in {processed_dir}. Ensure preprocessing was run and successful.")
        # Verify all masks exist
        missing_masks = [mp for mp in self.mask_paths if not os.path.exists(mp)]
        if missing_masks:
            print(f"Warning: {len(missing_masks)} mask files are missing for corresponding images in {processed_dir}. Example: {missing_masks[0] if missing_masks else 'N/A'}")
            # Filter out image paths that don't have a corresponding mask
            valid_indices = [i for i, mp in enumerate(self.mask_paths) if os.path.exists(mp)]
            self.img_paths = [self.img_paths[i] for i in valid_indices]
            self.mask_paths = [self.mask_paths[i] for i in valid_indices]
            print(f"Proceeding with {len(self.img_paths)} valid image-mask pairs after filtering.")


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img  = np.load(self.img_paths[idx]).astype(np.float32) # Shape (H, W, D), should be [0,1]
        mask = np.load(self.mask_paths[idx]).astype(np.int64)  # Shape (H, W, D)

        img_tensor  = torch.from_numpy(img).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img_tensor),
            mask =tio.LabelMap(tensor=mask_tensor),
        )
        if self.transform:
            subject = self.transform(subject)

        return subject.image.data, subject.mask.data.squeeze(0).long()

class PreprocessedDataset(Dataset):
    """Loads preprocessed .npy volumes from cached directories (for Val/Test)."""
    def __init__(self, data_info_list, return_original_path=False):
        self.data_info_list = data_info_list
        self.return_original_path = return_original_path
        if not self.data_info_list:
            print(f"Warning: PreprocessedDataset initialized with an empty data_info_list.")
        elif return_original_path:
            # Verify all items have 3 elements if original path is expected
            invalid_items = [item for item in self.data_info_list if len(item) != 3]
            if invalid_items:
                print(f"Warning: {len(invalid_items)} items in PreprocessedDataset (test mode) do not have 3 elements (img_path, mask_path, original_raw_path). Example malformed item: {invalid_items[0] if invalid_items else 'N/A'}")
                # This could lead to errors later if not handled by caller or if list becomes empty
                self.data_info_list = [item for item in self.data_info_list if len(item) == 3]
                print(f"Proceeding with {len(self.data_info_list)} valid 3-element items after filtering.")


    def __len__(self):
        return len(self.data_info_list)

    def __getitem__(self, idx):
        info = self.data_info_list[idx]
        npy_img_path = info[0]
        npy_mask_path = info[1]

        img  = np.load(npy_img_path).astype(np.float32)
        mask = np.load(npy_mask_path).astype(np.int64)

        img_tensor  = torch.from_numpy(img).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).long()

        if self.return_original_path:
            if len(info) < 3: # Should have been caught by init, but as a safeguard
                print(f"ERROR in PreprocessedDataset __getitem__ for test: item {idx} has {len(info)} elements, expected 3. Info: {info}")
                # Return dummy path or raise error to make it obvious something is wrong
                return img_tensor, mask_tensor, "ERROR_PATH_NOT_FOUND"
            original_raw_img_path = info[2]
            return img_tensor, mask_tensor, original_raw_img_path
        else:
            return img_tensor, mask_tensor
print("[NOTEBOOK_LOG] Dataset classes defined.")

# --- Model Definition (3D U-Net with Attention Gates) ---
print("[NOTEBOOK_LOG] Defining 3D U-Net model with Attention Gates...")
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )

class AttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi_input = self.relu(g1 + x1)
        alpha = self.psi(psi_input)
        return x * alpha.expand_as(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        # Encoder
        self.enc1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = conv_block(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.ag3 = AttentionGate3D(F_g=128, F_l=128, F_int=64)
        self.dec3 = conv_block(256, 128) # 128 (upsampled) + 128 (attended skip)

        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.ag2 = AttentionGate3D(F_g=64, F_l=64, F_int=32)
        self.dec2 = conv_block(128, 64) # 64 + 64

        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.ag1 = AttentionGate3D(F_g=32, F_l=32, F_int=16)
        self.dec1 = conv_block(64, 32) # 32 + 32

        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        enc1_out = self.enc1(x)
        pool1_out = self.pool1(enc1_out)
        enc2_out = self.enc2(pool1_out)
        pool2_out = self.pool2(enc2_out)
        enc3_out = self.enc3(pool2_out)
        pool3_out = self.pool3(enc3_out)
        bottleneck_out = self.bottleneck(pool3_out)

        up3_out = self.up3(bottleneck_out)
        attended_enc3_out = self.ag3(g=up3_out, x=enc3_out)
        dec3_in = torch.cat((up3_out, attended_enc3_out), dim=1)
        dec3_out = self.dec3(dec3_in)

        up2_out = self.up2(dec3_out)
        attended_enc2_out = self.ag2(g=up2_out, x=enc2_out)
        dec2_in = torch.cat((up2_out, attended_enc2_out), dim=1)
        dec2_out = self.dec2(dec2_in)

        up1_out = self.up1(dec2_out)
        attended_enc1_out = self.ag1(g=up1_out, x=enc1_out)
        dec1_in = torch.cat((up1_out, attended_enc1_out), dim=1)
        dec1_out = self.dec1(dec1_in)

        logits = self.out_conv(dec1_out)

        if DEBUG:
            print(f"Input x shape: {x.shape}")
            print(f"enc1_out: {enc1_out.shape}, pool1_out: {pool1_out.shape}")
            # ... (add other debug prints if needed)
            print(f"logits: {logits.shape}")
        return logits
print("[NOTEBOOK_LOG] Model definition complete.")


# --- Focal Tversky Loss ---
print("[NOTEBOOK_LOG] Defining Focal Tversky Loss...")
epoch_for_loss_debug_global = 0 # Use a distinct name for global variable
batch_idx_for_loss_debug_global = 0

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-5): # Increased smooth
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.epsilon = 1e-7 # Small constant for numerical stability

    def forward(self, preds_logits, targets_labels):
        preds_softmax = F.softmax(preds_logits, dim=1)
        targets_one_hot = F.one_hot(targets_labels, num_classes=NUM_CLASSES).permute(0, 4, 1, 2, 3).float()

        if DEBUG and epoch_for_loss_debug_global == 0 and batch_idx_for_loss_debug_global == 0 :
            print(f"Loss - preds_softmax shape: {preds_softmax.shape}, targets_one_hot shape: {targets_one_hot.shape}")

        dims = (2, 3, 4)
        TP = (preds_softmax * targets_one_hot).sum(dim=dims)
        FP = (preds_softmax * (1 - targets_one_hot)).sum(dim=dims)
        FN = ((1 - preds_softmax) * targets_one_hot).sum(dim=dims)

        denominator = TP + self.alpha * FN + self.beta * FP + self.smooth
        tversky_index = (TP + self.smooth) / (denominator + self.epsilon) # Add epsilon to denominator

        # Clip tversky_index to prevent (1 - tversky_index) from being negative before power
        tversky_index = torch.clamp(tversky_index, min=self.epsilon, max=1.0 - self.epsilon)

        focal_tversky_loss_per_class = torch.pow((1 - tversky_index), self.gamma)

        if torch.isnan(focal_tversky_loss_per_class).any() or torch.isinf(focal_tversky_loss_per_class).any():
            print(f"NaN/Inf detected in focal_tversky_loss_per_class at epoch {epoch_for_loss_debug_global}, batch {batch_idx_for_loss_debug_global}")
            print(f"  TP: {TP.min():.2e} to {TP.max():.2e}, FP: {FP.min():.2e} to {FP.max():.2e}, FN: {FN.min():.2e} to {FN.max():.2e}")
            print(f"  Denominator: {denominator.min():.2e} to {denominator.max():.2e}, Tversky Index: {tversky_index.min():.4f} to {tversky_index.max():.4f}")
            if DEBUG: # Only save if DEBUG is true to avoid filling disk space
                 torch.save({'preds': preds_logits.detach().cpu(), 'targets': targets_labels.detach().cpu(),
                            'TP':TP.detach().cpu(), 'FP':FP.detach().cpu(), 'FN':FN.detach().cpu(),
                            'tversky': tversky_index.detach().cpu()}, f'nan_debug_batch_e{epoch_for_loss_debug_global}_b{batch_idx_for_loss_debug_global}.pt')
            # Depending on severity, you might want to return a large valid loss or raise an error
            # For now, let it propagate to see if training recovers or consistently fails.

        return focal_tversky_loss_per_class.mean()

criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-5)
print("[NOTEBOOK_LOG] Loss function defined and instantiated.")

# --- Training and Evaluation ---
print("[NOTEBOOK_LOG] Defining training and evaluation utility functions (compute_metrics, preprocess_and_cache_subset)...")
def compute_metrics(pred_mask_np, target_mask_np, num_classes=NUM_CLASSES, smooth=1e-6):
    dice_scores = []
    jaccard_scores = []
    per_class_metrics = {}
    for cls in range(num_classes):
        pred_cls = (pred_mask_np == cls).astype(np.float32)
        target_cls = (target_mask_np == cls).astype(np.float32)

        intersection = np.sum(pred_cls * target_cls)
        pred_sum = np.sum(pred_cls)
        target_sum = np.sum(target_cls)

        dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
        union = pred_sum + target_sum - intersection
        jaccard = (intersection + smooth) / (union + smooth)

        dice_scores.append(dice)
        jaccard_scores.append(jaccard)
        per_class_metrics[f'class_{cls}'] = {'dice': dice, 'jaccard': jaccard}

    pixel_accuracy = np.mean(pred_mask_np.flatten() == target_mask_np.flatten())
    mean_dice_all_classes = np.mean(dice_scores)
    mean_jaccard_all_classes = np.mean(jaccard_scores)

    mean_dice_foreground = np.mean(dice_scores[1:]) if num_classes > 1 and len(dice_scores) > 1 else mean_dice_all_classes
    mean_jaccard_foreground = np.mean(jaccard_scores[1:]) if num_classes > 1 and len(jaccard_scores) > 1 else mean_jaccard_all_classes

    return per_class_metrics, mean_dice_all_classes, mean_jaccard_all_classes, pixel_accuracy, mean_dice_foreground, mean_jaccard_foreground


def preprocess_and_cache_subset(raw_subset_dir, cached_subset_dir, subset_name, is_test_set=False):
    print(f"[NOTEBOOK_LOG - {subset_name}] Top of preprocess_and_cache_subset. Checking cache: {cached_subset_dir}")
    os.makedirs(cached_subset_dir, exist_ok=True)

    data_info_list = []
    npy_img_files = sorted(glob.glob(os.path.join(cached_subset_dir, "*_img.npy")))
    basename_to_original_path = {}

    if is_test_set:
        temp_raw_image_mask_pairs = get_image_mask_pairs(raw_subset_dir)
        for img_path, _ in temp_raw_image_mask_pairs:
            basename_to_original_path[get_basename(img_path)] = img_path

    for npy_img_path in npy_img_files:
        npy_mask_path = npy_img_path.replace("_img.npy", "_mask.npy")
        if os.path.exists(npy_mask_path):
            if is_test_set:
                base_name_npy = get_basename(npy_img_path.replace("_img.npy", ""))
                original_raw_img_path = basename_to_original_path.get(base_name_npy)
                if original_raw_img_path:
                    data_info_list.append((npy_img_path, npy_mask_path, original_raw_img_path))
                else:
                    if DEBUG: print(f"Debug - {subset_name}: Could not find original path for cached test file {npy_img_path} (base: {base_name_npy})")
            else:
                data_info_list.append((npy_img_path, npy_mask_path))
        else:
            if DEBUG: print(f"Debug - {subset_name}: Mask pair not found for {npy_img_path} in cache.")

    is_cache_sufficient = False
    if npy_img_files: # Cache files exist
        if len(data_info_list) == len(npy_img_files): # All npy images have a corresponding mask and entry
            if is_test_set:
                is_cache_sufficient = all(len(item) == 3 for item in data_info_list)
                if not is_cache_sufficient and data_info_list : # Some items are not 3-tuples
                     print(f"Warning - {subset_name}: Cache has {len(npy_img_files)} images, {len(data_info_list)} pairs formed, but not all test items have 3 elements. Reprocessing.")
            else: # Train/Val
                is_cache_sufficient = True if data_info_list else False # Ensure list isn't empty if npy_files existed
        else: # Mismatch between found npy files and successfully paired items
            print(f"Warning - {subset_name}: Cache has {len(npy_img_files)} images, but only {len(data_info_list)} valid pairs found. Reprocessing.")
    else: # No npy files found in cache
        pass # is_cache_sufficient remains False


    if is_cache_sufficient:
        print(f"Loaded {len(data_info_list)} items from existing cache for {subset_name}.")
        return data_info_list
    elif npy_img_files: # Cache exists but is incomplete/problematic
        print(f"Cache for {subset_name} exists but is incomplete or problematic (e.g., test paths missing or mask pairs not found for all images). Reprocessing all for this subset.")
        data_info_list = [] # Clear partially populated list and reprocess everything
    else: # No .npy image files found at all
        print(f"Cache for {subset_name} is empty. Preprocessing raw data.")
        data_info_list = []

    print(f"[NOTEBOOK_LOG - {subset_name}] Preprocessing raw data from {raw_subset_dir}...")
    if not os.path.exists(raw_subset_dir):
        print(f"❌ Raw {subset_name} data directory not found: {raw_subset_dir}")
        return []

    print(f"[NOTEBOOK_LOG - {subset_name}] Calling get_image_mask_pairs for: {raw_subset_dir}")
    raw_image_mask_pairs = get_image_mask_pairs(raw_subset_dir)
    print(f"[NOTEBOOK_LOG - {subset_name}] get_image_mask_pairs found {len(raw_image_mask_pairs)} pairs.")

    if not raw_image_mask_pairs:
        print(f"❌ No image-mask pairs found to process in {raw_subset_dir} for {subset_name}. Preprocessing cannot continue.")
        return []

    for i, (img_path, mask_path) in enumerate(raw_image_mask_pairs):
        try:
            base_name = get_basename(img_path)
            if i < 2 or DEBUG:
                print(f"[NOTEBOOK_LOG - {subset_name}] Processing raw file {i+1}/{len(raw_image_mask_pairs)}: {base_name} from {img_path}")

            cached_img_filepath = os.path.join(cached_subset_dir, f"{base_name}_img.npy")
            cached_mask_filepath = os.path.join(cached_subset_dir, f"{base_name}_mask.npy")

            img = load_medical_image(img_path)
            img_p = preprocess_image_raw(img, INPUT_SHAPE, order=1)
            np.save(cached_img_filepath, img_p)

            mask = load_medical_image(mask_path)
            mask_p = preprocess_mask_raw(mask, INPUT_SHAPE, order=0)
            np.save(cached_mask_filepath, mask_p)

            if is_test_set:
                data_info_list.append((cached_img_filepath, cached_mask_filepath, img_path))
            else:
                data_info_list.append((cached_img_filepath, cached_mask_filepath))
        except Exception as e:
            print(f"Error processing {img_path} or {mask_path} for {subset_name} (file {i+1}): {e}")

    print(f"{subset_name} data preprocessing and caching complete. {len(data_info_list)} items processed into cache.")
    return data_info_list

print("[NOTEBOOK_LOG] Training and evaluation utility functions defined.")

def main():
    print("[NOTEBOOK_LOG] main() function initiated.")
    global epoch_for_loss_debug_global, batch_idx_for_loss_debug_global # For debugging loss

    if DEBUG:
        print("[NOTEBOOK_LOG] DEBUG mode is ON. Enabling autograd anomaly detection.")
        torch.autograd.set_detect_anomaly(True)

    print("--- Starting Data Preprocessing and Caching ---")
    train_data_info = preprocess_and_cache_subset(RAW_TRAIN_DIR, CACHED_TRAIN_DIR, "Train")
    val_data_info = preprocess_and_cache_subset(RAW_VAL_DIR, CACHED_VAL_DIR, "Validation")
    test_data_info = preprocess_and_cache_subset(RAW_TEST_DIR, CACHED_TEST_DIR, "Test", is_test_set=True)
    print("--- Data Preprocessing and Caching Finished ---\n")

    print("[NOTEBOOK_LOG] Initializing datasets...")
    train_dataset = TorchIODataset(CACHED_TRAIN_DIR, transform=train_transform)
    val_dataset = PreprocessedDataset(val_data_info)
    print("[NOTEBOOK_LOG] Datasets initialized.")

    if not len(train_dataset):
        print("❌ Training dataset is empty. Please check data paths and preprocessing logs. Exiting.")
        sys.exit(1) # Exit if no training data

    num_workers = 2 if DEVICE.type == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))
    print(f"[NOTEBOOK_LOG] Train DataLoader created. Num workers: {num_workers}, Train samples: {len(train_dataset)}")

    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))
        print(f"[NOTEBOOK_LOG] Validation DataLoader created. Val samples: {len(val_dataset)}")
    else:
        print(f"⚠️ Validation dataset is empty or failed to load.")

    print("[NOTEBOOK_LOG] Initializing U-Net model...")
    model = UNet3D(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=MIN_LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    print("[NOTEBOOK_LOG] Model, optimizer, scheduler, and GradScaler initialized.")

    print(f"\n--- Starting model training for {EPOCHS} epochs on {DEVICE} ---")
    start_time = time.time()
    best_val_dice = -1.0 # Initialize with a value lower than any possible dice

    for epoch in range(EPOCHS):
        epoch_for_loss_debug_global = epoch
        model.train()
        train_loss_epoch = 0.0

        print(f"[NOTEBOOK_LOG] Starting Epoch {epoch+1}/{EPOCHS} - Training phase...")
        for batch_idx, (imgs, masks) in enumerate(train_loader):
            batch_idx_for_loss_debug_global = batch_idx
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            autocast_context = torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda"))
            with autocast_context:
                outputs = model(imgs)
                loss = criterion(outputs, masks)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"💥 NaN/Inf loss detected at Epoch {epoch+1}, Batch {batch_idx+1}! Loss: {loss.item()}")
                print("  Skipping optimizer step for this batch.")
                # Potentially save problematic inputs/outputs here if not done by loss function
                # Consider stopping training or reducing LR drastically if this happens often
                if DEBUG:
                    torch.save({'imgs':imgs.cpu(), 'masks':masks.cpu(), 'outputs':outputs.cpu()},
                               f"nan_loss_epoch{epoch+1}_batch{batch_idx+1}.pt")
                continue # Skip update for this batch

            if DEVICE.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                scaler.step(optimizer)
                scaler.update()
            else: # For CPU
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                optimizer.step()

            train_loss_epoch += loss.item()

            if DEBUG and batch_idx == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Train Loss: {loss.item():.4f}")
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}: Current Avg Batch Loss: {train_loss_epoch/(batch_idx+1):.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

        avg_train_loss = train_loss_epoch / len(train_loader) if len(train_loader) > 0 else float('nan')
        print(f"[NOTEBOOK_LOG] Epoch {epoch+1} - Training complete. Avg Train Loss: {avg_train_loss:.4f}")

        # Validation phase
        val_loss_epoch = 0.0
        avg_val_loss = float('nan')
        val_dice_epoch_total_fg = 0.0
        num_val_samples_processed = 0

        if val_loader and len(val_dataset) > 0:
            print(f"[NOTEBOOK_LOG] Epoch {epoch+1} - Validation phase starting...")
            model.eval()
            with torch.no_grad():
                for val_batch_idx, (imgs_val, masks_val) in enumerate(val_loader):
                    imgs_val, masks_val = imgs_val.to(DEVICE), masks_val.to(DEVICE)

                    autocast_context_val = torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda"))
                    with autocast_context_val:
                        outputs_val = model(imgs_val)
                        loss_val = criterion(outputs_val, masks_val)

                    if torch.isnan(loss_val) or torch.isinf(loss_val):
                        print(f"💥 NaN/Inf validation loss detected at Epoch {epoch+1}, Val Batch {val_batch_idx+1}! Loss: {loss_val.item()}")
                        # If val loss is NaN, it's problematic. We might skip this batch's metrics or assign worst-case.
                        # For now, it adds to val_loss_epoch, which might make avg_val_loss NaN.
                    val_loss_epoch += loss_val.item()


                    probs_val = F.softmax(outputs_val, dim=1)
                    preds_np_val = torch.argmax(probs_val, dim=1).cpu().numpy()
                    masks_np_val = masks_val.cpu().numpy()

                    for i_val in range(preds_np_val.shape[0]):
                        _, _, _, _, m_dice_fg, _ = compute_metrics(
                            preds_np_val[i_val], masks_np_val[i_val], NUM_CLASSES
                        )
                        val_dice_epoch_total_fg += m_dice_fg
                        num_val_samples_processed +=1

                    if (val_batch_idx + 1) % 10 == 0 or (val_batch_idx + 1) == len(val_loader):
                        print(f"  Epoch {epoch+1}, Val Batch {val_batch_idx+1}/{len(val_loader)} processed.")

            if num_val_samples_processed > 0:
                avg_val_loss = val_loss_epoch / len(val_loader) if len(val_loader) > 0 else float('nan')
                avg_val_dice_fg = val_dice_epoch_total_fg / num_val_samples_processed

                print(f"Epoch {epoch+1}/{EPOCHS} -- Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val FG Dice: {avg_val_dice_fg:.4f}")

                scheduler.step(avg_val_dice_fg) # Step scheduler based on foreground dice

                if avg_val_dice_fg > best_val_dice and not (torch.isnan(torch.tensor(avg_val_dice_fg)) or torch.isinf(torch.tensor(avg_val_dice_fg))): # Ensure dice is valid
                    best_val_dice = avg_val_dice_fg
                    torch.save(model.state_dict(), os.path.join(OUTPUT_MASK_PATH, "best_attention_unet_model.pth"))
                    print(f"    🎉 Model saved (New best Val FG Dice: {best_val_dice:.4f})")
                else:
                    print(f"    💔 Val FG Dice ({avg_val_dice_fg:.4f}) did not improve from best ({best_val_dice:.4f}) or was NaN/Inf.")
            else:
                print(f"Epoch {epoch+1}/{EPOCHS} -- Train Loss: {avg_train_loss:.4f} (Validation loader present but processed 0 samples or Val Dice was NaN/Inf)")
                scheduler.step(metrics=0) # Or some default value if no valid metric. Or scheduler.step(avg_train_loss) if mode='min'
            print(f"[NOTEBOOK_LOG] Epoch {epoch+1} - Validation phase complete.")
        else: # No validation loader or empty validation set
            print(f"Epoch {epoch+1}/{EPOCHS} -- Train Loss: {avg_train_loss:.4f} (No validation data or loader)")
            scheduler.step(metrics=0) # Or step based on train_loss if configured for it
            if (epoch + 1) % 50 == 0 or epoch + 1 == EPOCHS :
                torch.save(model.state_dict(), os.path.join(OUTPUT_MASK_PATH, f"attention_unet_model_epoch_{epoch+1}.pth"))
                print(f"    💾 Model saved (no validation, epoch {epoch+1})")
        
        # Check if learning rate is at minimum and best_val_dice hasn't improved for a while (early stopping)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr <= (MIN_LR + 1e-9) and (epoch - np.argmax([best_val_dice if 'best_val_dice' in locals() else -1])) > (SCHEDULER_PATIENCE * 2) : # A bit arbitrary check
            print(f"Learning rate at minimum and no improvement for {SCHEDULER_PATIENCE * 2} epochs. Consider early stopping.")
            # break # Optional: uncomment to enable early stopping


    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds_val = divmod(rem, 60)
    print("\n--- Training Completed ---")
    print(f"Total Training Time: {int(hours)}h {int(minutes)}m {seconds_val:.2f}s")
    if val_loader and len(val_dataset) > 0 : print(f"Best validation foreground Dice achieved: {best_val_dice:.4f}")

    # --- Inference & Evaluation on Test Set ---
    print("\n[NOTEBOOK_LOG] Starting Test Set Inference & Evaluation...")
    model_path_to_load = os.path.join(OUTPUT_MASK_PATH, "best_attention_unet_model.pth")

    if not os.path.exists(model_path_to_load):
        print(f"Best model not found at {model_path_to_load}. Looking for latest epoch model...")
        epoch_models = glob.glob(os.path.join(OUTPUT_MASK_PATH, "attention_unet_model_epoch_*.pth"))
        if epoch_models:
            latest_epoch_num = -1
            path_to_latest_epoch_model = ""
            for em_path in epoch_models:
                try:
                    ep_num = int(os.path.basename(em_path).split('_')[-1].split('.')[0])
                    if ep_num > latest_epoch_num:
                        latest_epoch_num = ep_num
                        path_to_latest_epoch_model = em_path
                except ValueError: continue
            if path_to_latest_epoch_model:
                model_path_to_load = path_to_latest_epoch_model
                print(f"Loading model from latest available epoch: {model_path_to_load}")
            else:
                print(f"❌ No suitable epoch model found in {OUTPUT_MASK_PATH}. Skipping testing.")
                return
        else:
            print(f"❌ No saved models (best or epoch-based) found in {OUTPUT_MASK_PATH}. Skipping testing.")
            return
    else:
        print(f"Loading best model: {model_path_to_load}")

    try:
        # Attempt to load with weights_only=True for PyTorch 1.6+
        try:
            model.load_state_dict(torch.load(model_path_to_load, map_location=DEVICE, weights_only=True))
        except TypeError: # Fallback for older PyTorch
            print("Warning: 'weights_only' argument not supported by this PyTorch version. Loading without it.")
            model.load_state_dict(torch.load(model_path_to_load, map_location=DEVICE))
    except FileNotFoundError:
        print(f"❌ Model file not found at {model_path_to_load}. Skipping testing.")
        return
    except Exception as e:
        print(f"❌ Error loading model state_dict: {e}. Skipping testing.")
        return

    model.eval()

    if not test_data_info:
        print("❌ Test data info list is empty. Skipping testing.")
        return

    test_dataset = PreprocessedDataset(test_data_info, return_original_path=True)
    if not len(test_dataset):
        print("❌ Test dataset is empty (possibly due to filtering in PreprocessedDataset init). Skipping testing.")
        return
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    print(f"[NOTEBOOK_LOG] Test DataLoader created with {len(test_dataset)} samples.")

    all_dice_scores_fg = []
    all_jaccard_scores_fg = []
    all_pixel_accuracies = []
    all_per_class_metrics_results = [{'dice': [], 'jaccard': []} for _ in range(NUM_CLASSES)]

    print("\n--- Test Set Evaluation & Prediction Saving ---")
    for i, data_batch in enumerate(test_loader):
        if not (isinstance(data_batch, (list, tuple)) and len(data_batch) == 3):
            print(f"❌ Error: Test loader did not return expected 3 items (img, mask, path) at index {i}. Got {len(data_batch) if isinstance(data_batch, (list,tuple)) else type(data_batch)}. Skipping sample.")
            continue
        img_tensor, mask_gt_tensor, original_raw_image_path_tuple = data_batch

        if not isinstance(original_raw_image_path_tuple, tuple) or not original_raw_image_path_tuple or original_raw_image_path_tuple[0] == "ERROR_PATH_NOT_FOUND":
            print(f"❌ Error: original_raw_image_path_tuple is not a non-empty valid tuple at index {i}. Got: {original_raw_image_path_tuple}. Skipping sample.")
            continue
        original_raw_image_path = original_raw_image_path_tuple[0]

        img_tensor, mask_gt_tensor = img_tensor.to(DEVICE), mask_gt_tensor.to(DEVICE)

        with torch.no_grad():
            autocast_context_test = torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda"))
            with autocast_context_test:
                output_logits = model(img_tensor)
            probabilities = F.softmax(output_logits, dim=1)
            predicted_mask_tensor = torch.argmax(probabilities, dim=1)

        pred_mask_np = predicted_mask_tensor.cpu().numpy().squeeze()
        gt_mask_np = mask_gt_tensor.cpu().numpy().squeeze()

        per_class_metrics, _, _, px_acc, m_dice_fg, m_jaccard_fg = compute_metrics(
            pred_mask_np, gt_mask_np, NUM_CLASSES
        )
        all_dice_scores_fg.append(m_dice_fg)
        all_jaccard_scores_fg.append(m_jaccard_fg)
        all_pixel_accuracies.append(px_acc)

        for cls_idx in range(NUM_CLASSES):
            all_per_class_metrics_results[cls_idx]['dice'].append(per_class_metrics[f'class_{cls_idx}']['dice'])
            all_per_class_metrics_results[cls_idx]['jaccard'].append(per_class_metrics[f'class_{cls_idx}']['jaccard'])

        print(f"Processed test sample {i+1}/{len(test_dataset)}: {os.path.basename(original_raw_image_path)}")
        print(f"  FG Dice: {m_dice_fg:.4f}, FG Jaccard: {m_jaccard_fg:.4f}, Accuracy: {px_acc:.4f}")

        try:
            original_img_data_shape = None
            original_header_info = {}

            if original_raw_image_path.lower().endswith('.nrrd'):
                temp_data, original_header = nrrd.read(original_raw_image_path)
                original_img_data_shape = temp_data.shape
                original_header_info['nrrd_header'] = original_header
            elif original_raw_image_path.lower().endswith(('.nii.gz', '.nii')):
                nifti_img = nib.load(original_raw_image_path)
                original_img_data_shape = nifti_img.shape
                original_header_info['nifti_affine'] = nifti_img.affine
                original_header_info['nifti_header'] = nifti_img.header
            else:
                print(f"  Skipping resampling for unsupported file type: {original_raw_image_path}")

            if original_img_data_shape:
                zoom_factors = [original_img_data_shape[d] / pred_mask_np.shape[d] for d in range(pred_mask_np.ndim)]
                resampled_pred_mask_np = scipy.ndimage.zoom(pred_mask_np, zoom_factors, order=0, mode='nearest').astype(np.short)

                base_name_for_save = get_basename(original_raw_image_path)
                output_filename = os.path.join(OUTPUT_MASK_PATH, f"{base_name_for_save}_pred_resampled.nrrd")

                nrrd_header_to_save = {}
                if 'nrrd_header' in original_header_info:
                    nrrd_header_to_save = copy.deepcopy(original_header_info['nrrd_header'])
                    nrrd_header_to_save['type'] = resampled_pred_mask_np.dtype.name
                    nrrd_header_to_save['sizes'] = np.array(resampled_pred_mask_np.shape)
                    for key_to_remove in ['data file', ' Gesamtbildaufnahmezeitpunkt', 'keyvaluepairs', 'content', 'encoding', 'datafile', 'data file ']: # added more variants
                        if key_to_remove in nrrd_header_to_save: del nrrd_header_to_save[key_to_remove]
                        if key_to_remove.strip() in nrrd_header_to_save: del nrrd_header_to_save[key_to_remove.strip()]


                elif 'nifti_affine' in original_header_info:
                    affine = original_header_info['nifti_affine']
                    nrrd_header_to_save['type'] = resampled_pred_mask_np.dtype.name
                    nrrd_header_to_save['dimension'] = resampled_pred_mask_np.ndim
                    nrrd_header_to_save['space'] = 'left-posterior-superior'
                    nrrd_header_to_save['sizes'] = np.array(resampled_pred_mask_np.shape)
                    space_directions = affine[:resampled_pred_mask_np.ndim, :resampled_pred_mask_np.ndim]
                    nrrd_header_to_save['space directions'] = space_directions.tolist()
                    nrrd_header_to_save['space origin'] = affine[:resampled_pred_mask_np.ndim, resampled_pred_mask_np.ndim].tolist()
                    nrrd_header_to_save['kinds'] = ['domain'] * resampled_pred_mask_np.ndim
                    nrrd_header_to_save['endian'] = 'little'
                else: # Fallback minimal header
                    nrrd_header_to_save['type'] = resampled_pred_mask_np.dtype.name
                    nrrd_header_to_save['dimension'] = resampled_pred_mask_np.ndim
                    nrrd_header_to_save['sizes'] = np.array(resampled_pred_mask_np.shape)
                    nrrd_header_to_save['space'] = 'unknown' # Or some default
                    nrrd_header_to_save['kinds'] = ['domain'] * resampled_pred_mask_np.ndim


                nrrd_header_to_save['encoding'] = 'gzip'

                nrrd.write(output_filename, resampled_pred_mask_np, header=nrrd_header_to_save)
                print(f"  3D Slicer-compatible NRRD mask saved to: {output_filename}")
            else:
                print(f"  Original shape/header not determined. Saving prediction at INPUT_SHAPE.")
                fallback_filename = os.path.join(OUTPUT_MASK_PATH, f"{get_basename(original_raw_image_path)}_pred_INPUT_SHAPE.nrrd")
                nrrd.write(fallback_filename, pred_mask_np.astype(np.short))
                print(f"  Fallback: Saved non-resampled mask to {fallback_filename}")

        except Exception as e_save:
            print(f"  Error saving resampled mask for {original_raw_image_path}: {e_save}")
            fallback_filename_err = os.path.join(OUTPUT_MASK_PATH, f"{get_basename(original_raw_image_path)}_pred_INPUT_SHAPE_SAVE_ERROR.nrrd")
            try:
                nrrd.write(fallback_filename_err, pred_mask_np.astype(np.short))
                print(f"  Fallback (due to save error): Saved non-resampled mask to {fallback_filename_err}")
            except Exception as e_fb_save:
                print(f"  Critical error: Could not even save fallback for {original_raw_image_path}: {e_fb_save}")

    print("\n--- Overall Test Evaluation Metrics ---")
    if all_dice_scores_fg:
        print(f"Mean Foreground Dice Coefficient (averaged over samples): {np.mean(all_dice_scores_fg):.4f}")
        print(f"Mean Foreground Jaccard Index (IoU) (averaged over samples): {np.mean(all_jaccard_scores_fg):.4f}")
        print(f"Mean Pixel Accuracy (averaged over samples): {np.mean(all_pixel_accuracies):.4f}")

        print("\nPer-Class Test Metrics (averaged over samples):")
        for cls_idx in range(NUM_CLASSES):
            mean_cls_dice = np.mean(all_per_class_metrics_results[cls_idx]['dice']) if all_per_class_metrics_results[cls_idx]['dice'] else 0.0
            mean_cls_jaccard = np.mean(all_per_class_metrics_results[cls_idx]['jaccard']) if all_per_class_metrics_results[cls_idx]['jaccard'] else 0.0
            print(f"  Class {cls_idx}: Mean Dice = {mean_cls_dice:.4f}, Mean Jaccard = {mean_cls_jaccard:.4f}")
    else:
        print("No test samples were processed or metrics could not be calculated.")
    print("[NOTEBOOK_LOG] Test Set Inference & Evaluation finished.")


if __name__ == "__main__":
    print("[NOTEBOOK_LOG] Script execution entering __main__ block.")
    print("[NOTEBOOK_LOG] Calling main() function...")
    main()
    print("[NOTEBOOK_LOG] main() function call has returned. Script ending.")
