import torch
from tqdm import tqdm
from utils import center_crop_tensor


def global_dice_from_binary(pred_bin, target_bin, smooth=1e-5):
    pred_bin = pred_bin.float()
    target_bin = target_bin.float()
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return float(dice.item())


def search_best_threshold_global(pred_probs, target, thresholds=None):
    if thresholds is None:
        thresholds = torch.arange(0.30, 0.71, 0.02, device=pred_probs.device)

    best_thr = 0.5
    best_dice = 0.0

    for thr in thresholds:
        pred_bin = (pred_probs > thr).float()
        dice = global_dice_from_binary(pred_bin, target)
        if dice > best_dice:
            best_dice = dice
            best_thr = float(thr.item())

    return best_dice, best_thr


def evaluate_model(model, dataloader, criterion, device, model_name=None):
    model.eval()
    val_loss = 0.0
    all_probs = []
    all_masks = []

    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)

            if model_name == 'unet':
                outputs = center_crop_tensor(outputs, masks.size(2), masks.size(3))

            loss = criterion(outputs, masks)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            all_probs.append(probs)
            all_masks.append(masks)

    avg_loss = val_loss / len(dataloader)

    probs = torch.cat(all_probs, dim=0)
    masks = torch.cat(all_masks, dim=0)

    dice_05 = global_dice_from_binary((probs > 0.5).float(), masks)
    tuned_dice, best_thr = search_best_threshold_global(probs, masks)

    return {
        "avg_loss": avg_loss,
        "avg_dice_main": dice_05,
        "tuned_dice_main": tuned_dice,
        "best_thr_main": best_thr,
        "avg_dice_full": dice_05,
        "tuned_dice_full": tuned_dice,
        "best_thr_full": best_thr,
    }