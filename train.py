import os
import zipfile
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import PhysicsGuidedDUN, kernel_regularization_loss

# =====================================================================
# 1. Self-Contained Differentiable SSIM Loss (Zero-Dependency)
# =====================================================================

def gaussian_window(window_size, sigma):
    gauss = torch.tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_loss(img1, img2, window_size=11, size_average=True):
    """
    Computes differentiable SSIM loss: 1 - SSIM(img1, img2)
    """
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return 1.0 - ssim_map.mean()
    else:
        return 1.0 - ssim_map.mean(1).mean(1).mean(1)


# =====================================================================
# 2. Differentiable Edge (Sobel) Loss
# =====================================================================

def edge_loss(pred, target):
    """
    Penalizes difference in high-frequency gradients (boundaries) using Sobel filters.
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    
    # Mirror pad to handle image borders cleanly
    pred_pad = F.pad(pred, (1, 1, 1, 1), mode='reflect')
    target_pad = F.pad(target, (1, 1, 1, 1), mode='reflect')
    
    pred_gx = F.conv2d(pred_pad, sobel_x)
    pred_gy = F.conv2d(pred_pad, sobel_y)
    
    target_gx = F.conv2d(target_pad, sobel_x)
    target_gy = F.conv2d(target_pad, sobel_y)
    
    return F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)


# =====================================================================
# 3. Paired Wafer Dataset & Augmentations
# =====================================================================

class PairedWaferDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir, augment=True):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.augment = augment
        
        # Match filenames
        valid_exts = ('.npy', '.png', '.jpg', '.jpeg', '.tif', '.tiff')
        self.filenames = sorted([
            f for f in os.listdir(noisy_dir) 
            if f.lower().endswith(valid_exts) and os.path.exists(os.path.join(gt_dir, f))
        ])
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        name = self.filenames[idx]
        noisy_path = os.path.join(self.noisy_dir, name)
        gt_path = os.path.join(self.gt_dir, name)
        
        # Load degraded image
        ext = os.path.splitext(name)[1].lower()
        if ext == '.npy':
            noisy_arr = np.load(noisy_path)
            gt_arr = np.load(gt_path)
            
            noisy_tensor = torch.from_numpy(noisy_arr).float()
            gt_tensor = torch.from_numpy(gt_arr).float()
        else:
            from PIL import Image
            noisy_img = Image.open(noisy_path).convert('L')
            gt_img = Image.open(gt_path).convert('L')
            
            noisy_tensor = torch.from_numpy(np.array(noisy_img, dtype=np.float32) / 255.0)
            gt_tensor = torch.from_numpy(np.array(gt_img, dtype=np.float32) / 255.0)
            
        # Ensure (C, H, W) shapes
        if noisy_tensor.ndim == 2:
            noisy_tensor = noisy_tensor.unsqueeze(0)
        if gt_tensor.ndim == 2:
            gt_tensor = gt_tensor.unsqueeze(0)
            
        # Geometric augmentations (applied identically to pair)
        if self.augment:
            # 1. Random horizontal flip
            if random.random() > 0.5:
                noisy_tensor = torch.flip(noisy_tensor, dims=[2])
                gt_tensor = torch.flip(gt_tensor, dims=[2])
                
            # 2. Random vertical flip
            if random.random() > 0.5:
                noisy_tensor = torch.flip(noisy_tensor, dims=[1])
                gt_tensor = torch.flip(gt_tensor, dims=[1])
                
            # 3. Random transpose (diagonal rotation)
            if random.random() > 0.5:
                noisy_tensor = noisy_tensor.transpose(1, 2)
                gt_tensor = gt_tensor.transpose(1, 2)
                
        return noisy_tensor, gt_tensor


# =====================================================================
# 4. Training loop execution
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Physics-Guided DUN Training Script")
    parser.add_argument('--train_zip', default='train.zip', help='Path to KLA train.zip')
    parser.add_argument('--data_dir', default='./data', help='Extract location for data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--weights_dir', default='./checkpoints', help='Where to save weights')
    parser.add_argument('--channels', type=int, default=32, help='Model channels')
    parser.add_argument('--num_iterations', type=int, default=4, help='Unrolled iterations')
    parser.add_argument('--steps_per_df', type=int, default=3, help='Data Fidelity inner steps')
    parser.add_argument('--mixed_precision', action='store_true', help='Enable mixed precision')
    parser.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # 1. Automatically extract data if zip exists and extraction folder is missing
    if not os.path.exists(args.data_dir) and os.path.exists(args.train_zip):
        print(f"Extracting {args.train_zip} to {args.data_dir}...")
        with zipfile.ZipFile(args.train_zip, 'r') as zip_ref:
            zip_ref.extractall(args.data_dir)
        print("Extraction complete.")
        
    # Check paths
    noisy_dir = os.path.join(args.data_dir, 'train', 'NoisyLR')
    gt_dir = os.path.join(args.data_dir, 'train', 'GT')
    
    if not os.path.exists(noisy_dir) or not os.path.exists(gt_dir):
        # Fallback to local 'train' folder
        noisy_dir = './train/NoisyLR'
        gt_dir = './train/GT'
        
    if not os.path.exists(noisy_dir) or not os.path.exists(gt_dir):
        raise FileNotFoundError(f"Could not find training paths 'NoisyLR' or 'GT' in either {args.data_dir} or current folder.")
        
    print(f"Data directories loaded successfully:\n- NoisyLR: {noisy_dir}\n- Ground Truth: {gt_dir}")
    
    # 2. Build Dataset and Dataloaders (Split 85% train, 15% validation)
    full_dataset = PairedWaferDataset(noisy_dir, gt_dir, augment=True)
    num_samples = len(full_dataset)
    num_train = int(0.85 * num_samples)
    num_val = num_samples - num_train
    
    print(f"Total dataset size: {num_samples} pairs.")
    print(f"Split: Train={num_train} pairs, Validation={num_val} pairs.")
    
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, 
        [num_train, num_val], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Disable augmentations on the validation set
    val_set.dataset.augment = False
    
    pin_mem = True if device.type == 'cuda' else False
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)
    
    # 3. Model setup
    model = PhysicsGuidedDUN(
        num_iterations=args.num_iterations, 
        steps_per_df=args.steps_per_df, 
        channels=args.channels
    ).to(device)
    
    # 4. Optimizer and LR Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Mixed precision configuration
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
    
    # Loss weights configuration
    w_l1 = 1.0
    w_ssim = 0.5
    w_edge = 0.2
    w_kernel = 0.01  # Weight for blur kernel regularization
    
    best_val_loss = float('inf')
    os.makedirs(args.weights_dir, exist_ok=True)
    
    # 5. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_l1, train_ssim, train_edge, train_reg, train_total = 0.0, 0.0, 0.0, 0.0, 0.0
        
        optimizer.zero_grad()
        
        for batch_idx, (lr_imgs, gt_imgs) in enumerate(train_loader):
            lr_imgs = lr_imgs.to(device)
            gt_imgs = gt_imgs.to(device)
            
            # Forward pass with autocast
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                outputs = model(lr_imgs)
                
                # Compute loss elements
                loss_l1 = F.l1_loss(outputs, gt_imgs)
                loss_ssim = ssim_loss(outputs, gt_imgs)
                loss_edge = edge_loss(outputs, gt_imgs)
                loss_reg = kernel_regularization_loss(model.blur_op)
                
                # Total loss
                total_loss = w_l1 * loss_l1 + w_ssim * loss_ssim + w_edge * loss_edge + w_kernel * loss_reg
                
                # Scale loss for gradient accumulation
                total_loss_scaled = total_loss / args.grad_accum
            
            # Backpropagation
            scaler.scale(total_loss_scaled).backward()
            
            # Step optimizer every grad_accum steps
            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            # Accumulate logs
            train_l1 += loss_l1.item()
            train_ssim += loss_ssim.item()
            train_edge += loss_edge.item()
            train_reg += loss_reg.item()
            train_total += total_loss.item()
            
        # 6. Validation Loop
        model.eval()
        val_l1, val_ssim, val_edge, val_total = 0.0, 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for lr_imgs, gt_imgs in val_loader:
                lr_imgs = lr_imgs.to(device)
                gt_imgs = gt_imgs.to(device)
                
                with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                    outputs = model(lr_imgs)
                    
                    loss_l1 = F.l1_loss(outputs, gt_imgs)
                    loss_ssim = ssim_loss(outputs, gt_imgs)
                    loss_edge = edge_loss(outputs, gt_imgs)
                    
                    total_loss = w_l1 * loss_l1 + w_ssim * loss_ssim + w_edge * loss_edge
                
                val_l1 += loss_l1.item()
                val_ssim += loss_ssim.item()
                val_edge += loss_edge.item()
                val_total += total_loss.item()
                
        # Normalize logs
        num_train_batches = len(train_loader)
        num_val_batches = len(val_loader)
        
        print(f"Epoch [{epoch}/{args.epochs}]")
        print(f"  Train -> Loss: {train_total/num_train_batches:.4f} | L1: {train_l1/num_train_batches:.4f} | SSIM_L: {train_ssim/num_train_batches:.4f} | Edge: {train_edge/num_train_batches:.4f} | Reg: {train_reg/num_train_batches:.4f}")
        print(f"  Val   -> Loss: {val_total/num_val_batches:.4f} | L1: {val_l1/num_val_batches:.4f} | SSIM_L: {val_ssim/num_val_batches:.4f} | Edge: {val_edge/num_val_batches:.4f}")
        
        scheduler.step()
        
        # Save checkpoints
        avg_val_loss = val_total / num_val_batches
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(args.weights_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_path)
            print(f"  [RECORD] New validation record! Saved best checkpoint to {best_path}")
            
        # Periodic saving
        if epoch % 10 == 0:
            epoch_path = os.path.join(args.weights_dir, f'model_epoch_{epoch}.pt')
            torch.save(model.state_dict(), epoch_path)
            print(f"  Checkpoint saved: {epoch_path}")
            
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
