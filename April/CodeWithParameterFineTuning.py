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
import optuna # Import Optuna
import torch.optim.lr_scheduler as lr_scheduler # Import LR schedulers
import torch.nn.utils as nn_utils # Import utility for gradient clipping
import json # Import json for saving raw trials

# --- Configuration ---
INPUT_SHAPE = (128, 128, 64)     # Resize images to this shape
# BATCH_SIZE = 2 # Will be tuned or set within objective - Using FIXED_BATCH_SIZE
# EPOCHS = 500 # Use a fixed number for trials, maybe fewer, then retrain best
# LEARNING_RATE = 1e-4 # Will be tuned
NUM_CLASSES = 4              # (class 0 = background, classes 1..3 = foreground)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False                # Set to True for debug prints/visualizations

# --- Training Stability Parameters ---
MAX_GRAD_NORM = 1.0          # Maximum norm for gradient clipping
EARLY_STOP_PATIENCE = 20     # Number of epochs to wait for validation loss improvement
MIN_DELTA = 1e-4             # Minimum change in validation loss to qualify as improvement

# --- Learning Rate Scheduler Parameters ---
# Using ReduceLROnPlateau: Reduce LR when validation loss has stopped improving.
LR_SCHEDULER_MODE = 'min'    # Minimize the monitored quantity (validation loss)
LR_SCHEDULER_FACTOR = 0.5    # Factor by which the learning rate will be reduced. new_lr = lr * factor
LR_SCHEDULER_PATIENCE = 10   # Number of epochs with no improvement after which learning rate will be reduced.
LR_SCHEDULER_MIN_LR = 1e-7   # A scalar. A lower bound on the learning rate.

# --- Optuna Configuration ---
N_TRIALS = 25  # Number of hyperparameter combinations to try
EPOCHS_PER_TRIAL = 50 # Number of epochs to train for each trial (adjust for speed vs accuracy)
FINAL_TRAINING_EPOCHS = 1000 # Epochs to train the final model with best params (can be original 500)
FIXED_BATCH_SIZE = 2 # Keep batch size fixed for simplicity, or tune it too

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
# Output directory for predictions and reports
# Consider updating this name if you rerun, e.g., include date/time or a version number
OUTPUT_MASK_PATH = os.path.join(TEST_DIR, "predictionsCUDATest_Optimized31Apr")
os.makedirs(OUTPUT_MASK_PATH, exist_ok=True) # Ensure output directory exists

# --- Utility Functions for Raw Data (Mostly unchanged) ---
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
    # Normalize intensity
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    # Resize
    zoom_factors = [shape[i] / img.shape[i] for i in range(3)]
    return scipy.ndimage.zoom(img, zoom_factors, order=order)

def preprocess_mask(mask, shape=INPUT_SHAPE, order=0):
    # Resize using nearest neighbor interpolation
    zoom_factors = [shape[i] / mask.shape[i] for i in range(3)]
    return scipy.ndimage.zoom(mask, zoom_factors, order=order)

def get_image_mask_pairs(directory):
    image_files = glob.glob(os.path.join(directory, "images", "*"))
    mask_files = glob.glob(os.path.join(directory, "masks", "*"))

    def get_basename(file_path):
        base = os.path.basename(file_path)
        # Adjusted regex to be more robust potentially
        base = re.sub(r'\((SCAN|MASK|Mask)\)_cropped\.nrrd$', '', base, flags=re.IGNORECASE).strip()
        base = re.sub(r'\.nrrd$|\.nii\.gz$|\.nii$', '', base, flags=re.IGNORECASE).strip() # Handle other extensions too
        return base

    image_dict = {get_basename(f): f for f in image_files}
    mask_dict = {get_basename(f): f for f in mask_files}

    common_keys = set(image_dict.keys()) & set(mask_dict.keys())
    images = [image_dict[key] for key in sorted(common_keys)] # Sort for consistency
    masks = [mask_dict[key] for key in sorted(common_keys)]

    if len(common_keys) == 0:
        print(f"❌ No matching images and masks found in {directory}. Check filenames and get_basename function!")
        # Debugging: Print keys if no match
        # print("Image keys:", sorted(image_dict.keys()))
        # print("Mask keys:", sorted(mask_dict.keys()))
    else:
        print(f"🔍 Found {len(images)} image-mask pairs in {directory}")
    return images, masks

# --- Dataset Classes (Unchanged) ---
class CachedMedicalDataset(Dataset):
    def __init__(self, processed_dir):
        self.image_paths = sorted(glob.glob(os.path.join(processed_dir, "*_img.npy")))
        self.mask_paths = [p.replace("_img.npy", "_mask.npy") for p in self.image_paths]
        if not self.image_paths or not all(os.path.exists(p) for p in self.mask_paths):
             print(f"Warning: Issue finding cached data in {processed_dir}. Ensure preprocessing script ran correctly.")
             print(f"Found {len(self.image_paths)} image paths.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = np.load(self.image_paths[idx])
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) # Add channel dim: (1, D, H, W) assuming numpy shape is (D, H, W)
            mask = np.load(self.mask_paths[idx])
            mask = torch.tensor(mask, dtype=torch.long) # Shape (D, H, W)
            return img, mask
        except Exception as e:
            print(f"Error loading cached data item {idx}: {self.image_paths[idx]}")
            print(e)
            # Return dummy data or raise error? Returning dummy might hide issues.
            # Let's return None and handle it in the loader or raise
            raise # Re-raise the exception

class MedicalDataset(Dataset):
    def __init__(self, directory):
        self.image_paths, self.mask_paths = get_image_mask_pairs(directory)
        if not self.image_paths:
             raise ValueError(f"No image-mask pairs found in {directory}. Cannot create dataset.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = load_medical_image(self.image_paths[idx])
            img = preprocess_image(img, INPUT_SHAPE, order=1) # Use the existing preprocessing
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) # Add channel dim: (1, D, H, W)

            mask = load_medical_image(self.mask_paths[idx])
            mask = preprocess_mask(mask, INPUT_SHAPE, order=0) # Use the existing preprocessing
            mask = torch.tensor(mask, dtype=torch.long) # Shape (D, H, W)
            return img, mask
        except Exception as e:
            print(f"Error loading raw data item {idx}: {self.image_paths[idx]}")
            print(e)
            raise # Re-raise the exception

# --- Model Definition (3D U-Net - Unchanged) ---
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), # Bias=False often used with BatchNorm
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
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
        self.dec3 = conv_block(256, 128) # Input channels: 128 (from up3) + 128 (from enc3 skip)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128, 64) # Input channels: 64 (from up2) + 64 (from enc2 skip)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(64, 32) # Input channels: 32 (from up1) + 32 (from enc1 skip)
        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        p1 = self.pool1(enc1)
        enc2 = self.enc2(p1)
        p2 = self.pool2(enc2)
        enc3 = self.enc3(p2)
        p3 = self.pool3(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(p3)

        # Decoder
        up3 = self.up3(bottleneck)
        # Ensure spatial dimensions match before concatenation if necessary
        # This assumes pooling/convtranspose maintain compatibility with skip connections
        cat3 = torch.cat((up3, enc3), dim=1)
        dec3 = self.dec3(cat3)

        up2 = self.up2(dec3)
        cat2 = torch.cat((up2, enc2), dim=1)
        dec2 = self.dec2(cat2)

        up1 = self.up1(dec2)
        cat1 = torch.cat((up1, enc1), dim=1)
        dec1 = self.dec1(cat1)

        # Output
        logits = self.out_conv(dec1)

        if DEBUG:
             print("Forward Pass Shapes:")
             print(f"Input: {x.shape}")
             print(f"enc1: {enc1.shape}, p1: {p1.shape}")
             print(f"enc2: {enc2.shape}, p2: {p2.shape}")
             print(f"enc3: {enc3.shape}, p3: {p3.shape}")
             print(f"bottleneck: {bottleneck.shape}")
             print(f"up3: {up3.shape}, cat3: {cat3.shape}, dec3: {dec3.shape}")
             print(f"up2: {up2.shape}, cat2: {cat2.shape}, dec2: {dec2.shape}")
             print(f"up1: {up1.shape}, cat1: {cat1.shape}, dec1: {dec1.shape}")
             print(f"logits: {logits.shape}")
        return logits


# --- Focal Tversky Loss (Unchanged) ---
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        # Ensure alpha and beta are within reasonable bounds if tuned
        self.alpha = max(0.01, min(0.99, alpha)) # Clamp alpha
        self.beta = 1.0 - self.alpha # Automatically set beta based on alpha
        # self.beta = beta # Or allow independent tuning, but alpha+beta=1 is common
        self.gamma = gamma
        self.smooth = smooth
        # print(f"Initialized FocalTverskyLoss with alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}")


    def forward(self, preds, targets):
        # Ensure preds are logits and targets are class indices (long)
        # print(f"Loss input shapes: preds={preds.shape}, targets={targets.shape}") # Debug shapes
        # print(f"Loss input types: preds={preds.dtype}, targets={targets.dtype}") # Debug types
        # print(f"Targets min/max: {targets.min()}, {targets.max()}") # Check target range

        if preds.shape[1] != NUM_CLASSES:
             raise ValueError(f"Prediction tensor channel dimension ({preds.shape[1]}) does not match NUM_CLASSES ({NUM_CLASSES})")

        preds = F.softmax(preds, dim=1)

        # One-hot encode targets: (N, D, H, W) -> (N, D, H, W, C) -> (N, C, D, H, W)
        try:
            targets_one_hot = F.one_hot(targets, num_classes=NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
        except RuntimeError as e:
            print(f"Error during one-hot encoding. Check if targets contain values outside [0, {NUM_CLASSES-1}].")
            print(f"Targets min/max: {targets.min()}, {targets.max()}")
            raise e


        # Spatial dimensions are 2, 3, 4 (D, H, W)
        dims = (2, 3, 4)
        TP = (preds * targets_one_hot).sum(dim=dims)
        FP = (preds * (1 - targets_one_hot)).sum(dim=dims)
        FN = ((1 - preds) * targets_one_hot).sum(dim=dims)

        Tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)

        # Handle potential division by zero or unstable values if TP/FN/FP sums are near zero
        # Although smooth factor helps, clamp Tversky index to avoid issues with power
        Tversky = torch.clamp(Tversky, min=self.smooth, max=1.0 - self.smooth)

        focal_tversky_loss = torch.pow(1 - Tversky, self.gamma)

        # Average loss over batch and classes
        return focal_tversky_loss.mean()

# --- Metrics Computation (Unchanged) ---
def compute_metrics(pred, target, num_classes=NUM_CLASSES, smooth=1e-6):
    dice_scores = []
    jaccard_scores = []
    per_class_metrics = {}

    # Ensure pred and target are numpy arrays
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    pred = pred.flatten()
    target = target.flatten()

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = np.sum(pred_cls * target_cls)
        pred_sum = np.sum(pred_cls)
        target_sum = np.sum(target_cls)

        # Avoid division by zero if a class is completely absent in prediction or target
        dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
        jaccard = (intersection + smooth) / (pred_sum + target_sum - intersection + smooth)

        dice_scores.append(dice)
        jaccard_scores.append(jaccard)
        per_class_metrics[f'class_{cls}'] = {'dice': dice, 'jaccard': jaccard}

    pixel_accuracy = np.mean(pred == target) # Direct comparison is simpler
    mean_dice = np.mean(dice_scores) # Average Dice over all classes
    mean_jaccard = np.mean(jaccard_scores) # Average Jaccard over all classes

    return per_class_metrics, mean_dice, mean_jaccard, pixel_accuracy


# --- Optuna Objective Function ---
def objective(trial, train_loader, val_loader):
    """Optuna objective function for hyperparameter tuning."""

    # --- Suggest Hyperparameters ---
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    # Tune optimizer type (optional)
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    optimizer_name = "Adam" # Keep Adam for now

    # Tune FocalTverskyLoss parameters
    # Ensure alpha + beta = 1 by tuning alpha
    alpha = trial.suggest_float("ft_alpha", 0.3, 0.7)
    # beta = 1.0 - alpha # Automatically set beta
    # Or tune beta independently if desired:
    # beta = trial.suggest_float("ft_beta", 0.3, 0.7)
    gamma = trial.suggest_float("ft_gamma", 0.5, 2.5) # Wider range for gamma maybe

    # --- Initialize Model, Optimizer, Loss, Scheduler ---
    model = UNet3D(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
    criterion = FocalTverskyLoss(alpha=alpha, beta=(1.0-alpha), gamma=gamma) # Use suggested params

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
         optimizer = optim.AdamW(model.parameters(), lr=lr)
    # Add other optimizers if tuning them

    # Learning Rate Scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               mode=LR_SCHEDULER_MODE,
                                               factor=LR_SCHEDULER_FACTOR,
                                               patience=LR_SCHEDULER_PATIENCE,
                                               min_lr=LR_SCHEDULER_MIN_LR)

    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # Early Stopping for the trial
    best_val_loss_trial = float('inf')
    epochs_no_improve_trial = 0
    trial_early_stop = False # Flag to break the epoch loop

    print(f"\n--- Trial {trial.number} ---")
    print(f"Params: LR={lr:.6f}, Optimizer={optimizer_name}, FT_Alpha={alpha:.4f}, FT_Gamma={gamma:.4f}")
    print(f"LR Scheduler: ReduceLROnPlateau (patience={LR_SCHEDULER_PATIENCE}, factor={LR_SCHEDULER_FACTOR})")
    print(f"Gradient Clipping: Max Norm = {MAX_GRAD_NORM}")
    print(f"Trial Early Stopping: Patience = {EARLY_STOP_PATIENCE}")


    # --- Training & Validation Loop for the Trial ---
    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        train_loss = 0.0
        for batch_idx, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad(set_to_none=True) # More efficient zeroing

            # Mixed precision context
            context = torch.amp.autocast(device_type=DEVICE.type, dtype=torch.float16) if DEVICE.type == "cuda" else nullcontext()

            with context:
                outputs = model(imgs)
                loss = criterion(outputs, masks)

            # Handle potential NaN loss immediately
            if torch.isnan(loss).any():
                 print(f"Warning: NaN loss detected in Trial {trial.number}, Epoch {epoch+1}, Batch {batch_idx}. Skipping batch.")
                 # Option 1: Skip this batch (continue inner loop)
                 # continue
                 # Option 2: Stop the trial early (might indicate bad hyperparameters)
                 print(f"Stopping Trial {trial.number} early due to NaN loss.")
                 trial.set_user_attr("NaN_loss_epoch", epoch + 1)
                 raise optuna.TrialPruned("NaN loss detected") # Prune trial with a reason
                 # Option 3: Adjust parameters and try to recover (more complex)


            # Scale loss
            scaler.scale(loss).backward()

            # Unscale before clipping
            scaler.unscale_(optimizer)

            # Gradient Clipping
            nn_utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
        train_loss /= (batch_idx + 1) # Average loss per batch

        # Handle case where train_loss might be NaN if all batches were skipped
        if np.isnan(train_loss):
             print(f"Warning: NaN train loss in Trial {trial.number}, Epoch {epoch+1}. Stopping trial.")
             trial.set_user_attr("NaN_train_loss_epoch", epoch + 1)
             raise optuna.TrialPruned("NaN train loss detected")


        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (imgs, masks) in enumerate(val_loader):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                context = torch.amp.autocast(device_type=DEVICE.type, dtype=torch.float16) if DEVICE.type == "cuda" else nullcontext()
                with context:
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)

                # Handle potential NaN validation loss
                if torch.isnan(loss).any():
                     print(f"Warning: NaN validation loss detected in Trial {trial.number}, Epoch {epoch+1}, Batch {batch_idx}. Stopping trial.")
                     trial.set_user_attr("NaN_val_loss_epoch", epoch + 1)
                     raise optuna.TrialPruned("NaN validation loss detected")

                val_loss += loss.item()
        val_loss /= (batch_idx + 1) # Average loss per batch

        # Handle case where val_loss might be NaN
        if np.isnan(val_loss):
             print(f"Warning: NaN val loss in Trial {trial.number}, Epoch {epoch+1}. Stopping trial.")
             trial.set_user_attr("NaN_val_loss_epoch", epoch + 1)
             raise optuna.TrialPruned("NaN val loss detected")


        print(f"Trial {trial.number} - Epoch {epoch+1}/{EPOCHS_PER_TRIAL} -- Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # --- Scheduler Step ---
        scheduler.step(val_loss)

        # --- Trial Early Stopping Check ---
        if val_loss < best_val_loss_trial - MIN_DELTA:
            best_val_loss_trial = val_loss
            epochs_no_improve_trial = 0
        else:
            epochs_no_improve_trial += 1

        if epochs_no_improve_trial >= EARLY_STOP_PATIENCE:
             print(f"Trial {trial.number}: Early stopping triggered at epoch {epoch+1} due to no validation loss improvement for {EARLY_STOP_PATIENCE} epochs.")
             trial_early_stop = True
             break # Exit epoch loop

        # --- Optuna Pruning (optional but recommended for efficiency) ---
        # Prune based on current validation loss after scheduler step
        trial.report(val_loss, epoch)
        if trial.should_prune():
             print(f"Trial {trial.number} pruned by Optuna at epoch {epoch+1}.")
             raise optuna.TrialPruned()

    # Return the metric to minimize (the best validation loss achieved)
    return best_val_loss_trial


# --- Main Execution ---
def main():
    # --- Load Data (Once) ---
    # Use cached training data
    try:
        train_dataset = CachedMedicalDataset(CACHED_TRAIN_DIR)
        # Validation and test are processed on the fly from raw data
        val_dataset = MedicalDataset(VAL_DIR)
        test_dataset = MedicalDataset(TEST_DIR)
    except ValueError as e:
         print(f"Error creating datasets: {e}")
         return # Exit if datasets can't be loaded

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: Training or Validation dataset is empty. Check data paths and preprocessing.")
        return

    # On Windows, num_workers=0 to avoid worker issues; on Linux, you can increase it.
    num_workers = 0 if os.name == 'nt' else 4
    train_loader = DataLoader(train_dataset, batch_size=FIXED_BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=FIXED_BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers) # Test with batch size 1 usually

    # --- Hyperparameter Optimization ---
    print("\n--- Starting Hyperparameter Optimization with Optuna ---")
    study = optuna.create_study(direction="minimize",
                                 pruner=optuna.pruners.MedianPruner())
    objective_with_data = lambda trial: objective(trial, train_loader, val_loader)
    start_opt_time = time.time()

    try:
        # a) Run all trials
        study.optimize(objective_with_data, n_trials=N_TRIALS)

        # Construct full paths for saving reports within the output directory
        optuna_csv_path = os.path.join(OUTPUT_MASK_PATH, "optuna_trials_report.csv")
        optuna_json_path = os.path.join(OUTPUT_MASK_PATH, "optuna_trials_report.json")
        optuna_history_html_path = os.path.join(OUTPUT_MASK_PATH, "optuna_optimization_history.html")
        optuna_importances_html_path = os.path.join(OUTPUT_MASK_PATH, "optuna_param_importances.html")


        # b) Dump trials to CSV
        trials_df = study.trials_dataframe(attrs=("number","value","params","state"))
        trials_df.to_csv(optuna_csv_path, index=False)
        print(f"✅ Saved trial report to {optuna_csv_path}")

        # c) Dump raw JSON
        trials_json = [t.__dict__ for t in study.trials]
        with open(optuna_json_path, "w") as f:
            json.dump(trials_json, f, default=str, indent=2)
        print(f"✅ Saved raw JSON report to {optuna_json_path}")

        # d) Visualizations
        import optuna.visualization as vis

        try:
            fig_hist = vis.plot_optimization_history(study)
            fig_hist.write_html(optuna_history_html_path)
            print(f"✅ Saved optimization history to {optuna_history_html_path}")
        except Exception as plot_err:
             print(f"Warning: Could not generate history plot. Error: {plot_err}")

        try:
            fig_imp = vis.plot_param_importances(study)
            fig_imp.write_html(optuna_importances_html_path)
            print(f"✅ Saved parameter importances to {optuna_importances_html_path}")
        except Exception as plot_err:
             print(f"Warning: Could not generate importance plot. Error: {plot_err}")


    except Exception as e:
        print(f"\nAn error occurred during Optuna optimization: {e}")
        import traceback
        traceback.print_exc()
        # If at least one trial completed, show its info
        try:
            best = study.best_trial
            print(f"\nBest trial so far: #{best.number} with value={best.value:.4f}")
            print("Params:", best.params)
        except ValueError:
            print("No successful trials to report.")
        # Decide whether to proceed to final training or exit.
        # If no trials completed, exit. If some completed, proceed with the best found so far.
        if len(study.trials) == 0 or study.best_trial is None:
             print("No successful trials. Exiting.")
             return # Exit if no trials completed successfully


    end_opt_time = time.time()
    print(f"\n--- Optuna Optimization Finished in {end_opt_time - start_opt_time:.1f}s ---")
    # Ensure best_trial is accessible even if an exception occurred after first trial
    try:
        best_trial = study.best_trial
        print(f"Best trial #{best_trial.number} → val_loss={best_trial.value:.4f}")
        print("Best hyperparameters:")
        best_params = best_trial.params
        for k,v in best_params.items():
            print(f"  • {k}: {v}")
    except ValueError:
        print("Could not retrieve best trial information after optimization.")
        # Attempt to load parameters from a specific trial if needed, or exit.
        # For simplicity, let's assume if we reached here, best_trial exists or an earlier error handler exited.
        return # Exit if best_trial is not available

    # --- Final Training with Best Hyperparameters ---
    print("\n--- Starting Final Training with Best Hyperparameters ---")
    final_lr = best_params["lr"]
    # final_optimizer_name = best_params.get("optimizer", "Adam") # Use get if optimizer was tuned
    final_optimizer_name = "Adam"
    final_alpha = best_params["ft_alpha"]
    final_beta = 1.0 - final_alpha # Consistent with loss definition
    final_gamma = best_params["ft_gamma"]

    final_model = UNet3D(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
    final_criterion = FocalTverskyLoss(alpha=final_alpha, beta=final_beta, gamma=final_gamma)

    if final_optimizer_name == "Adam":
        final_optimizer = optim.Adam(final_model.parameters(), lr=final_lr)
    elif final_optimizer_name == "AdamW":
         final_optimizer = optim.AdamW(final_model.parameters(), lr=final_lr)

    # Final Training Scheduler
    final_scheduler = lr_scheduler.ReduceLROnPlateau(final_optimizer,
                                                     mode=LR_SCHEDULER_MODE,
                                                     factor=LR_SCHEDULER_FACTOR,
                                                     patience=LR_SCHEDULER_PATIENCE,
                                                     min_lr=LR_SCHEDULER_MIN_LR)

    final_scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # Early Stopping for Final Training
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None # To save the best model weights

    start_train_time = time.time()

    for epoch in range(FINAL_TRAINING_EPOCHS):
        final_model.train()
        train_loss = 0.0
        for batch_idx, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            final_optimizer.zero_grad(set_to_none=True)
            context = torch.amp.autocast(device_type=DEVICE.type, dtype=torch.float16) if DEVICE.type == "cuda" else nullcontext()
            with context:
                outputs = final_model(imgs)
                loss = final_criterion(outputs, masks)

            # Handle potential NaN loss immediately in final training
            if torch.isnan(loss).any():
                 print(f"Fatal Error: NaN loss detected in Final Training, Epoch {epoch+1}, Batch {batch_idx}. Stopping training.")
                 # Decide how to handle this: break, exit, or try to recover.
                 # Breaking is a reasonable default to prevent further issues.
                 # Optionally, you could save the model state *before* this point if available.
                 break # Exit the epoch loop

            final_scaler.scale(loss).backward()

            # Unscale before clipping
            final_scaler.unscale_(final_optimizer)

            # Gradient Clipping
            nn_utils.clip_grad_norm_(final_model.parameters(), MAX_GRAD_NORM)

            final_scaler.step(final_optimizer)
            final_scaler.update()
            train_loss += loss.item()

        # Check for NaN train loss after batch loop
        if np.isnan(train_loss):
             print(f"Fatal Error: NaN train loss detected in Final Training, Epoch {epoch+1}. Stopping training.")
             break # Exit epoch loop

        train_loss /= (batch_idx + 1) # Average loss per batch (only if batch_idx >= 0)


        # Validation during final training
        final_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (imgs, masks) in enumerate(val_loader):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                context = torch.amp.autocast(device_type=DEVICE.type, dtype=torch.float16) if DEVICE.type == "cuda" else nullcontext()
                with context:
                    outputs = final_model(imgs)
                    loss = final_criterion(outputs, masks)

                # Handle potential NaN validation loss in final training
                if torch.isnan(loss).any():
                     print(f"Fatal Error: NaN validation loss detected in Final Training, Epoch {epoch+1}, Batch {batch_idx}. Stopping training.")
                     val_loss = float('inf') # Set to infinity to trigger early stopping check/prevent saving
                     break # Exit validation batch loop

                val_loss += loss.item()

        # Check for NaN val loss after batch loop
        if np.isnan(val_loss):
             print(f"Fatal Error: NaN val loss detected in Final Training, Epoch {epoch+1}. Stopping training.")
             break # Exit epoch loop

        # Only compute average val loss if batches were processed
        if batch_idx >= 0:
             val_loss /= (batch_idx + 1)
        else:
             # Handle case where val_loader might be empty, set val_loss accordingly
             val_loss = float('inf') if len(val_loader) > 0 else 0.0


        print(f"Final Training Epoch {epoch+1}/{FINAL_TRAINING_EPOCHS} -- Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {final_optimizer.param_groups[0]['lr']:.6f}")


        # --- Scheduler Step ---
        # Step the scheduler only if val_loss is a valid number
        if not np.isnan(val_loss):
             final_scheduler.step(val_loss)
        else:
             print(f"Warning: Skipping scheduler step due to NaN validation loss in Epoch {epoch+1}.")


        # --- Final Training Early Stopping Check ---
        # Only check for early stopping if val_loss is a valid number
        if not np.isnan(val_loss):
            if val_loss < best_val_loss - MIN_DELTA:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save the model state dict
                best_model_state = copy.deepcopy(final_model.state_dict())
                print(f"--> Saved best model state at epoch {epoch+1} with Val Loss: {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                 print(f"\n--- Final Training Early stopping triggered at epoch {epoch+1}. ---")
                 # Break the epoch loop *after* checking early stopping
                 break # Exit epoch loop
        else:
             print(f"Warning: Skipping early stopping check due to NaN validation loss in Epoch {epoch+1}.")
             # Consider if you want to stop training immediately on NaN val loss


    # Load the best model state if early stopping occurred and a state was saved
    if best_model_state is not None:
        final_model.load_state_dict(best_model_state)
        print("Loaded best model state based on validation performance.")
    elif epoch == FINAL_TRAINING_EPOCHS - 1: # If training finished all epochs without improvement
         print("Finished all epochs. No validation loss improvement found, using model from last epoch.")
         # The model state is already the one from the last epoch
    else: # If early stopping triggered before saving any state (shouldn't happen if best_val_loss starts high)
         print("Early stopping triggered, but no better model state was saved. This indicates no improvement from initial state.")
         # The model state is the one from the epoch the loop broke at.


    end_train_time = time.time()
    elapsed_time = end_train_time - start_train_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    print("\n--- Final Training Completed ---")
    print(f"Total Final Training Time: {elapsed_time:.2f} seconds (≈ {hours}h {minutes}m {seconds:.2f}s)")

    # --- Inference & Evaluation on Test Set with Final Model ---
    print("\n--- Evaluating Final Model on Test Set ---")
    final_model.eval()
    dice_list = []
    iou_scores = []
    accuracies = []
    per_class_results_list = [] # List to store dicts from compute_metrics for each test sample

    # Get test file pairs once
    test_image_paths, test_mask_paths = get_image_mask_pairs(TEST_DIR)
    if not test_image_paths:
         print("Error: No test image-mask pairs found. Cannot perform evaluation.")
         return

    # Use the test_loader for batch processing if possible (even B=1)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers) # Already defined earlier

    with torch.no_grad():
        for i, (imgs, masks_gt) in enumerate(test_loader): # Enumerate to match index with file paths if needed
            imgs, masks_gt = imgs.to(DEVICE), masks_gt.to(DEVICE) # masks_gt are ground truth here
            context = torch.amp.autocast(device_type=DEVICE.type, dtype=torch.float16) if DEVICE.type == "cuda" else nullcontext()

            with context:
                 output = final_model(imgs) # Shape (B, C, D, H, W)
                 probabilities = F.softmax(output, dim=1)
                 predicted_mask = torch.argmax(probabilities, dim=1) # Shape (B, D, H, W)

            # Move to CPU and compute metrics (assuming B=1 for test_loader)
            pred_np = predicted_mask.cpu().numpy().squeeze(0) # Shape (D, H, W)
            gt_np = masks_gt.cpu().numpy().squeeze(0)         # Shape (D, H, W)

            # --- Compute Metrics for the current test sample ---
            sample_metrics, sample_mean_dice, sample_mean_jaccard, sample_pixel_accuracy = compute_metrics(pred_np, gt_np, NUM_CLASSES)
            per_class_results_list.append(sample_metrics) # Store per-class results for this sample
            dice_list.append(sample_mean_dice) # Store mean dice for this sample
            iou_scores.append(sample_mean_jaccard) # Store mean Jaccard for this sample
            accuracies.append(sample_pixel_accuracy) # Store pixel accuracy for this sample

            # --- Save Predicted Mask ---
            # Get original image path corresponding to this batch item
            # Need to find the original path from the dataset list based on index i
            # Assuming test_dataset __getitem__ returns items in the same order as test_image_paths list
            if i < len(test_image_paths):
                 original_img_path = test_image_paths[i]
                 basename = os.path.basename(original_img_path)
            else:
                 # Fallback if index goes out of bounds (shouldn't happen with correct DataLoader setup)
                 print(f"Warning: Index {i} out of bounds for test_image_paths ({len(test_image_paths)}). Skipping mask saving.")
                 basename = f"unknown_sample_{i}" # Use a dummy name
                 original_img_path = None


            if original_img_path: # Only proceed if we found the original path
                 # Determine output filename
                 if basename.endswith('.nrrd'):
                     output_filename = os.path.join(OUTPUT_MASK_PATH, basename.replace(".nrrd", "_pred_optimized.nrrd"))
                 elif basename.endswith('.nii.gz'):
                      output_filename = os.path.join(OUTPUT_MASK_PATH, basename.replace(".nii.gz", "_pred_optimized.nrrd"))
                 elif basename.endswith('.nii'):
                      output_filename = os.path.join(OUTPUT_MASK_PATH, basename.replace(".nii", "_pred_optimized.nrrd"))
                 else:
                     output_filename = os.path.join(OUTPUT_MASK_PATH, basename + "_pred_optimized.nrrd")

                 try:
                     # Load original header info (from image or mask, assuming they match)
                     original_header = None
                     try:
                         if original_img_path.endswith('.nrrd'):
                             _, original_header = nrrd.read(original_img_path)
                         elif original_img_path.endswith(('.nii.gz', '.nii')):
                              img_nib = nib.load(original_img_path)
                              # Attempt to convert relevant nibabel header info to a dict for nrrd.write
                              original_header = {}
                              if hasattr(img_nib.header, 'get_zooms'):
                                   try:
                                        # Check if spacing is valid and 3D
                                        spacings = img_nib.header.get_zooms()[:3]
                                        if len(spacings) == 3 and all(s > 0 for s in spacings):
                                             original_header['spacings'] = list(spacings)
                                        else:
                                             print(f"Warning: Invalid or non-3D spacing {spacings} in header for {basename}. Skipping spacing in output header.")
                                   except Exception as sp_err:
                                        print(f"Warning: Could not get spacing from nibabel header for {basename}: {sp_err}")
                              # Add other relevant fields if needed, e.g., 'origin' (requires more complex conversion)
                         else:
                             print(f"Warning: Unsupported original file format for header reading: {basename}. Creating minimal header.")
                             original_header = {} # Fallback to empty dict

                         if not isinstance(original_header, dict): # Ensure it's a dict
                              print(f"Warning: Original header for {basename} is not a dictionary ({type(original_header)}). Creating minimal header.")
                              original_header = {}


                     except Exception as header_err:
                          print(f"Warning: Could not load header for {basename}: {header_err}. Creating minimal header.")
                          original_header = {}


                     # Resample prediction back to original image shape if possible
                     # Need to load original image data *again* just to get the shape reliably
                     original_data_for_shape = None
                     try:
                         original_data_for_shape = load_medical_image(original_img_path)
                     except Exception as load_err:
                         print(f"Warning: Could not load original image data for shape determination for {basename}: {load_err}")
                         original_data_for_shape = None # Ensure it's None if loading failed

                     if original_data_for_shape is not None:
                         orig_shape = original_data_for_shape.shape
                         if len(orig_shape) == 4: # Handle 4D NIfTI (e.g., time series) - use first 3 dims
                              orig_shape = orig_shape[:3]
                         if len(orig_shape) != 3:
                              print(f"Warning: Original image shape {original_data_for_shape.shape} for {basename} not 3D. Cannot resample prediction accurately.")
                              resampled_pred_mask = pred_np # Save prediction in its processed shape
                         elif pred_np.shape == orig_shape:
                              resampled_pred_mask = pred_np # No resampling needed
                              # print(f"Prediction shape {pred_np.shape} matches original {orig_shape} for {basename}. No resampling.")
                         else:
                             # print(f"Resampling prediction from {pred_np.shape} to original shape {orig_shape} for {basename}")
                             zoom_factors = [orig_shape[i] / pred_np.shape[i] for i in range(3)]
                             resampled_pred_mask = scipy.ndimage.zoom(pred_np, zoom_factors, order=0, mode='nearest') # Nearest neighbor for masks, use nearest mode for boundary
                     else:
                          print(f"Warning: Could not determine original image shape for {basename}. Saving prediction in processed shape {pred_np.shape}.")
                          resampled_pred_mask = pred_np # Save in processed shape


                     # Prepare header for saving prediction
                     new_header = copy.deepcopy(original_header) if isinstance(original_header, dict) else {}

                     # Ensure header is suitable for nrrd.write (must be dict)
                     if not isinstance(new_header, dict):
                          print("Warning: Original header was not a dict. Creating minimal header for NRRD.")
                          new_header = {}

                     new_header['type'] = 'short' # Or other appropriate type like 'unsigned char'
                     # Clean up problematic keys if they exist
                     if 'data file' in new_header: del new_header['data file']
                     if 'encoding' in new_header: del new_header['encoding']

                     # Update sizes if necessary based on the potentially resampled mask shape
                     new_header['sizes'] = np.array(resampled_pred_mask.shape)
                     if 'space dimension' in new_header and len(resampled_pred_mask.shape) != new_header['space dimension']:
                          new_header['space dimension'] = len(resampled_pred_mask.shape) # Update dimension


                     nrrd.write(output_filename, resampled_pred_mask.astype(np.int16), header=new_header)
                     print(f"Processed {i+1}/{len(test_loader)}. Saved prediction for {basename}")

                 except Exception as e:
                      print(f"Error processing or saving predicted mask for {basename}: {e}")
                      import traceback
                      traceback.print_exc()


        # --- Compute and Print Average Metrics ---
        # Calculate average metrics across all test samples
        avg_mean_dice = np.mean(dice_list) if dice_list else 0
        avg_mean_jaccard = np.mean(iou_scores) if iou_scores else 0
        avg_pixel_accuracy = np.mean(accuracies) if accuracies else 0

        # Calculate average per-class metrics
        # This requires aggregating the per_class_results_list
        avg_per_class_metrics = {}
        if per_class_results_list:
            for cls in range(NUM_CLASSES):
                class_dice_scores = [sample_metrics[f'class_{cls}']['dice'] for sample_metrics in per_class_results_list if f'class_{cls}' in sample_metrics]
                class_jaccard_scores = [sample_metrics[f'class_{cls}']['jaccard'] for sample_metrics in per_class_results_list if f'class_{cls}' in sample_metrics]

                # Handle cases where a class might be missing in all samples
                avg_per_class_metrics[f'class_{cls}'] = {
                    'avg_dice': np.mean(class_dice_scores) if class_dice_scores else 0,
                    'avg_jaccard': np.mean(class_jaccard_scores) if class_jaccard_scores else 0
                }

        print("\n--- Test Set Evaluation Results ---")
        print(f"Average Mean Dice across samples: {avg_mean_dice:.4f}")
        print(f"Average Mean Jaccard across samples: {avg_mean_jaccard:.4f}")
        print(f"Overall Pixel Accuracy across samples: {avg_pixel_accuracy:.4f}")

        print("\nAverage Per-Class Metrics:")
        for cls, metrics in avg_per_class_metrics.items():
             print(f"  {cls}:")
             print(f"    Avg Dice: {metrics['avg_dice']:.4f}")
             print(f"    Avg Jaccard: {metrics['avg_jaccard']:.4f}")

        # Log these results to a file within the output directory
        results_filename = os.path.join(OUTPUT_MASK_PATH, "test_evaluation_metrics.txt")
        with open(results_filename, "w") as f:
            f.write("--- Test Set Evaluation Results ---\n")
            f.write(f"Average Mean Dice across samples: {avg_mean_dice:.4f}\n")
            f.write(f"Average Mean Jaccard across samples: {avg_mean_jaccard:.4f}\n")
            f.write(f"Overall Pixel Accuracy across samples: {avg_pixel_accuracy:.4f}\n")
            f.write("\nAverage Per-Class Metrics:\n")
            for cls, metrics in avg_per_class_metrics.items():
                f.write(f"  {cls}:\n")
                f.write(f"    Avg Dice: {metrics['avg_dice']:.4f}\n")
                f.write(f"    Avg Jaccard: {metrics['avg_jaccard']:.4f}\n")
        print(f"Test evaluation metrics saved to {results_filename}")


    print("\n--- Inference and Evaluation Complete ---")


if __name__ == "__main__":
    main()
