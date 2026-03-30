import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from oxford_pet import OxfordPetDataset
from evaluate import evaluate_model
from utils import (
    EarlyStopping,
    plot_curves,
    BCEDiceLoss,
    save_validation_predictions,
    set_seed,
    seed_worker,
    center_crop_tensor,
)
import argparse
from tqdm import tqdm

def train(model_name,epochs=300,batch_size=24,lr=3e-4,seed=42,bce_weight=0.3,aug_scale_min=0.9,aug_scale_max=1.1,aug_angle_min=-15.0,aug_angle_max=15.0):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_name} on {device} (seed={seed})")

    pad_for_unet = (model_name == 'unet')

    train_dataset = OxfordPetDataset(
        'dataset/oxford-iiit-pet',
        'dataset/train.txt',
        is_train=True,
        aug_scale_range=(aug_scale_min, aug_scale_max),
        aug_angle_range=(aug_angle_min, aug_angle_max),
        pad_for_unet=pad_for_unet,
        has_mask=True,
    )
    val_dataset = OxfordPetDataset(
        'dataset/oxford-iiit-pet',
        'dataset/val.txt',
        is_train=False,
        pad_for_unet=pad_for_unet,
        has_mask=True,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    if model_name == 'unet':
        model = UNet().to(device)
    else:
        model = ResNet34_UNet().to(device)

    criterion = BCEDiceLoss(bce_weight=bce_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    early_stopping = EarlyStopping(patience=20)

    train_losses, val_losses, val_dices = [], [], []
    best_dice = 0.0
    best_threshold = 0.5

    os.makedirs('saved_models', exist_ok=True)
    best_model_path = f'saved_models/{model_name}_seed{seed}_best.pth'
    best_meta_path = f'saved_models/{model_name}_seed{seed}_best_meta.pth'

    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0.0
        model.train()

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for images, masks, _ in train_pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)

            if model_name == 'unet':
                outputs = center_crop_tensor(outputs, masks.size(2), masks.size(3))
            # outputs should already match masks size: [B,1,320,320]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)

        val_result = evaluate_model(
            model, val_loader, criterion, device, model_name=model_name
        )

        avg_val_loss = val_result["avg_loss"]
        avg_val_dice_main = val_result["avg_dice_main"]
        tuned_val_dice_main = val_result["tuned_dice_main"]
        tuned_thr_main = val_result["best_thr_main"]

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_dices.append(avg_val_dice_main)

        print(
            f"Epoch {epoch+1}/{epochs} | Time: {int(epoch_mins)}m {int(epoch_secs)}s | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Kaggle-like Val Dice@0.5: {avg_val_dice_main:.4f} | "
            f"Kaggle-like Val Dice@best({tuned_thr_main:.2f}): {tuned_val_dice_main:.4f}"
        )

        vis_dir = f'training_visualizations/{model_name}/seed{seed}'
        if epoch == 0 or (epoch + 1) % 10 == 0:
            save_validation_predictions(
                model,
                val_loader,
                device,
                epoch + 1,
                vis_dir,
                model_name=model_name,
                threshold=tuned_thr_main
            )

        scheduler.step()

        if tuned_val_dice_main > best_dice:
            best_dice = tuned_val_dice_main
            best_threshold = tuned_thr_main

            torch.save(model.state_dict(), best_model_path)
            torch.save({'threshold': best_threshold, 'seed': seed}, best_meta_path)

            # legacy filenames for inference compatibility
            torch.save(model.state_dict(), f'saved_models/{model_name}_best.pth')
            torch.save(
                {'threshold': best_threshold, 'seed': seed},
                f'saved_models/{model_name}_best_meta.pth'
            )

            print(
                f"Saved new best model with Dice: {best_dice:.4f} "
                f"at threshold {best_threshold:.2f} -> {best_model_path}"
            )

        # early stopping on main metric
        early_stopping(-tuned_val_dice_main)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    curve_dir = f'training_visualizations/{model_name}/seed{seed}'
    plot_curves(
        train_losses,
        val_losses,
        val_dices,
        model_name,
        suffix=f'seed{seed}',
        save_dir=curve_dir
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['unet', 'resnet34_unet'])
    parser.add_argument('--batch_size', type=int, default=24, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--bce_weight', type=float, default=0.3, help='Weight of BCE in BCE+Dice loss')
    parser.add_argument('--aug_scale_min', type=float, default=0.9, help='Min random affine scale for train aug')
    parser.add_argument('--aug_scale_max', type=float, default=1.1, help='Max random affine scale for train aug')
    parser.add_argument('--aug_angle_min', type=float, default=-15.0, help='Min random affine angle for train aug')
    parser.add_argument('--aug_angle_max', type=float, default=15.0, help='Max random affine angle for train aug')
    args = parser.parse_args()

    print(
        f"Args => model: {args.model}, batch_size: {args.batch_size}, epochs: {args.epochs}, "
        f"lr: {args.lr}, seed: {args.seed}, bce_weight: {args.bce_weight}, "
        f"aug_scale: [{args.aug_scale_min}, {args.aug_scale_max}], "
        f"aug_angle: [{args.aug_angle_min}, {args.aug_angle_max}]"
    )

    train(
        args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        bce_weight=args.bce_weight,
        aug_scale_min=args.aug_scale_min,
        aug_scale_max=args.aug_scale_max,
        aug_angle_min=args.aug_angle_min,
        aug_angle_max=args.aug_angle_max,
    )