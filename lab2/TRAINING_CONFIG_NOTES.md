# Oxford Pet Binary Segmentation Training Notes

This note is synced with the current code under src.

## 1) Pipeline Overview

- Task: binary segmentation (pet foreground vs background)
- Dataset loader: src/oxford_pet.py
- Training entry: src/train.py
- Validation logic: src/evaluate.py
- Inference and submission: src/inference.py

## 2) Default Training Configuration

- Input base size: 320 x 320
- Models: unet, resnet34_unet
- Batch size: 24
- Epochs: 300
- LR: 3e-4
- Optimizer: AdamW (weight_decay=1e-4)
- Scheduler: CosineAnnealingLR(T_max=epochs, eta_min=1e-6)
- Loss: BCEDiceLoss (default bce_weight=0.3)
- Early stopping: patience=20, monitored on tuned validation Dice (higher is better)

## 3) Data Processing and Augmentation

From src/oxford_pet.py:

- Image resize: bilinear
- Mask resize: nearest
- Label mapping:
  - trimap 1 -> foreground 1
  - trimap 2/3 -> background 0
- Normalize with ImageNet mean/std
- Train augmentation:
  - random horizontal flip
  - random affine
    - angle default: [-15, 15]
    - translation: up to 10 percent
    - scale default: [0.9, 1.1]
  - random brightness/contrast

UNet-only preprocessing:

- If model is unet, dataset uses reflect padding before model input.
- Current pad width is 94 pixels each side.
- Goal: keep original valid-conv UNet while aligning output to 320 x 320 evaluation space.

## 4) Model-Specific Shape Handling

UNet path:

- Original valid-conv architecture (no BN, no conv padding in model).
- In training/evaluation/inference, output is center-cropped to match mask space when needed.

ResNet34_UNet path:

- Uses native 320 x 320 input/output flow.
- Does not use UNet reflect padding branch.

## 5) Validation Metric

From src/evaluate.py:

- Collect logits/probabilities on validation set.
- Compute Dice at threshold 0.5.
- Search best threshold on range [0.30, 0.70) with step 0.02.
- Return:
  - avg_loss
  - avg_dice_main
  - tuned_dice_main
  - best_thr_main

Checkpoint selection in train.py uses tuned_dice_main.

## 6) Saved Artifacts

- saved_models/<model>_seed<seed>_best.pth
- saved_models/<model>_seed<seed>_best_meta.pth
- saved_models/<model>_best.pth (legacy single-model filename)
- saved_models/<model>_best_meta.pth (stores threshold and seed)

## 7) Inference Behavior

From src/inference.py:

- Load best checkpoint and best threshold.
- Optional scale TTA (argument --tta_scales).
- Horizontal flip TTA per scale.
- Average logits across TTA branches.
- Resize to original image size.
- Sigmoid + threshold + RLE encode.
- Save submission_<model>.csv.

UNet inference uses the same reflect-padding idea as training for consistency.

## 8) Ensemble Policy (Important)

This assignment forbids ensemble submission.

- Do not average multiple seeds/models in final submission.
- Do not add ensemble CLI/options.
- Current code is single-model inference only and does not include seed ensemble flow.

## 9) Practical Experiment Notes

- Compare experiments with the same metric definition and threshold policy.
- Keep train/eval/inference preprocessing consistent per model.
- For UNet, verify reflect padding and center-crop are both active and consistent.
- For ResNet34_UNet, keep standard 320 x 320 pipeline.

## 10) Command Examples

Train UNet:

python src/train.py --model unet

Train ResNet34_UNet:

python src/train.py --model resnet34_unet

Inference UNet:

python src/inference.py --model unet --test_txt dataset/test_unet.txt --tta_scales 1.0

Inference ResNet34_UNet:

python src/inference.py --model resnet34_unet --test_txt dataset/test_res_unet.txt --tta_scales 1.0