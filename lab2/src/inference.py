import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from oxford_pet import OxfordPetDataset
from utils import rle_encode, center_crop_tensor
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
import argparse

def create_model(model_name, device):
    if model_name == 'unet':
        return UNet().to(device)
    return ResNet34_UNet().to(device)


def load_model(model_name, device):
    model = create_model(model_name, device)
    weight_path = f'saved_models/{model_name}_best.pth'
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    threshold = 0.5
    meta_path = f'saved_models/{model_name}_best_meta.pth'
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, map_location='cpu')
        threshold = float(meta.get('threshold', 0.5))

    print(f"Using tuned threshold: {threshold:.2f}")
    print("Loading weights from:", weight_path)
    print("Loading meta from:", meta_path)
    return model, threshold


def maybe_pad_for_unet(images, model_name):
    if model_name != 'unet':
        return images
    # same reflect padding as training/validation preprocessing
    return F.pad(images, (94, 94, 94, 94), mode='reflect')


def inference(model_name, test_txt, tta_scales):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, threshold = load_model(model_name, device)

    # No mask required for inference
    test_dataset = OxfordPetDataset(
        'dataset/oxford-iiit-pet',
        test_txt,
        is_train=False,
        pad_for_unet=False,   # we pad dynamically per TTA scale
        has_mask=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    results = []

    print(f"Running inference for {model_name}...")
    print(f"Using scale TTA: {tta_scales}")

    with torch.no_grad():
        for images, _, file_names in tqdm(test_loader):
            images = images.to(device)
            _, _, h, w = images.shape

            tta_outputs = []
            for scale in tta_scales:
                scaled_h = max(32, int(round(h * scale)))
                scaled_w = max(32, int(round(w * scale)))

                scaled_images = F.interpolate(
                    images,
                    size=(scaled_h, scaled_w),
                    mode='bilinear',
                    align_corners=False
                )

                # original
                scaled_images_model = maybe_pad_for_unet(scaled_images, model_name)
                outputs_orig = model(scaled_images_model)
                if model_name == 'unet':
                    outputs_orig = center_crop_tensor(outputs_orig, scaled_h, scaled_w)

                # horizontal flip
                images_flipped = torch.flip(scaled_images, dims=[3])
                images_flipped_model = maybe_pad_for_unet(images_flipped, model_name)
                outputs_flipped = model(images_flipped_model)
                if model_name == 'unet':
                    outputs_flipped = center_crop_tensor(outputs_flipped, scaled_h, scaled_w)
                outputs_flipped_back = torch.flip(outputs_flipped, dims=[3])

                outputs_scale = (outputs_orig + outputs_flipped_back) / 2.0

                # bring back to base 320x320 eval space
                outputs_scale = F.interpolate(
                    outputs_scale,
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                )
                tta_outputs.append(outputs_scale)

            outputs = torch.stack(tta_outputs, dim=0).mean(dim=0)

            # resize back to original image resolution for submission
            orig_img_path = os.path.join('dataset/oxford-iiit-pet/images', file_names[0] + '.jpg')
            orig_w, orig_h = Image.open(orig_img_path).size
            outputs_resized = F.interpolate(
                outputs,
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=False
            )

            preds = torch.sigmoid(outputs_resized)
            pred_mask = (preds.squeeze().cpu().numpy() > threshold).astype(np.uint8)

            rle = rle_encode(pred_mask)
            results.append({'image_id': file_names[0], 'encoded_mask': rle})

    df = pd.DataFrame(results)
    csv_name = f'submission_{model_name}.csv'
    df.to_csv(csv_name, index=False)
    print(f"Saved Kaggle submission to {csv_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['unet', 'resnet34_unet'])
    parser.add_argument('--test_txt', type=str, required=True)
    parser.add_argument('--tta_scales', type=str, default='1.0')
    args = parser.parse_args()

    tta_scales = [float(s.strip()) for s in args.tta_scales.split(',') if s.strip()]
    inference(args.model, args.test_txt, tta_scales)