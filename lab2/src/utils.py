import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def dice_score(pred, target):
    smooth = 1e-5
    pred = (pred > 0.5).float()

    pred = pred.reshape(pred.size(0), -1)
    target = target.reshape(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    return dice.mean()

def best_dice_threshold(pred_probs, target, thresholds=None):
    if thresholds is None:
        thresholds = torch.arange(0.3, 0.61, 0.02, device=pred_probs.device)

    best_thr = 0.5
    best_dice = 0.0
    for thr in thresholds:
        dice = dice_score(pred_probs > thr, target).item()
        if dice > best_dice:
            best_dice = dice
            best_thr = float(thr.item())
    return best_dice, best_thr

def rle_encode(mask):
    """Encodes a mask in Run-Length Encoding (Fortran order)."""
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        
        # Dice on each sample then mean is more stable for segmentation batches.
        preds = torch.sigmoid(inputs)
        smooth = 1e-5
        preds = preds.reshape(preds.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        intersection = (preds * targets).sum(dim=1)
        dice = (2.0 * intersection + smooth) / (preds.sum(dim=1) + targets.sum(dim=1) + smooth)
        dice_loss = 1.0 - dice.mean()
        
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
    
def plot_curves(train_losses, val_losses, dice_scores, model_name, suffix=None, save_dir='.'):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, dice_scores, 'g-', label='Val Dice')
    plt.title('Validation Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filename = f'{model_name}_learning_curves.png' if suffix is None else f'{model_name}_{suffix}_learning_curves.png'
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def center_crop_tensor(x, target_h, target_w):
    _, _, h, w = x.shape
    start_y = (h - target_h) // 2
    start_x = (w - target_w) // 2
    return x[:, :, start_y:start_y + target_h, start_x:start_x + target_w]


def save_validation_predictions(
    model, val_loader, device, epoch, save_dir,
    model_name=None, num_samples=2, threshold=0.5
):

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    images, masks, _ = next(iter(val_loader))
    images = images[:num_samples].to(device)
    masks_full = masks[:num_samples]

    with torch.no_grad():
        outputs = model(images)

        # NEW: align unet output to 320x320 first
        if model_name == 'unet':
            outputs = center_crop_tensor(outputs, masks_full.size(2), masks_full.size(3))

        preds = torch.sigmoid(outputs)

        # eval-aligned (now same size as masks_full)
        pred_eval = (preds.squeeze(1).cpu().numpy() > threshold).astype(np.uint8)
        gt_eval_np = masks_full.squeeze(1).cpu().numpy().astype(np.uint8)

        # full-size reference
        preds_full = F.interpolate(
            preds,
            size=(masks_full.size(2), masks_full.size(3)),
            mode='bilinear',
            align_corners=False
        )
        pred_full = (preds_full.squeeze(1).cpu().numpy() > threshold).astype(np.uint8)
        gt_full = masks_full.squeeze(1).cpu().numpy().astype(np.uint8)

    # eval-aligned
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(num_samples):
        axes[i, 0].imshow(pred_eval[i], vmin=0, vmax=1)
        axes[i, 0].set_title(f"pred eval @thr={threshold:.2f}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt_eval_np[i], vmin=0, vmax=1)
        axes[i, 1].set_title("GT eval")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch:03d}_eval.png'))
    plt.close()

    # full-size reference
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(num_samples):
        axes[i, 0].imshow(pred_full[i], vmin=0, vmax=1)
        axes[i, 0].set_title(f"pred full @thr={threshold:.2f}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt_full[i], vmin=0, vmax=1)
        axes[i, 1].set_title("GT full")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch:03d}_full.png'))
    plt.close()