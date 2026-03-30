import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np
import random


class OxfordPetDataset(Dataset):
    def __init__(
        self,
        data_dir,
        txt_file,
        img_size=(320, 320),
        is_train=True,
        aug_scale_range=(0.9, 1.1),
        aug_angle_range=(-15.0, 15.0),
        pad_for_unet=False,
        has_mask=True,
        unet_pad=0,
    ):
        self.data_dir = data_dir
        self.img_size = img_size
        self.is_train = is_train
        self.aug_scale_range = aug_scale_range
        self.aug_angle_range = aug_angle_range
        self.pad_for_unet = pad_for_unet
        self.has_mask = has_mask

        self.img_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "annotations", "trimaps")

        # Original valid-conv UNet with 4 down/up blocks:
        # output_size = input_size - 188
        # To get output 320x320, input should be 508x508 => pad 94 on each side.
        self.unet_pad = 94

        with open(txt_file, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.file_names)

    def _load_mask(self, file_name):
        mask_path = os.path.join(self.mask_dir, file_name + '.png')
        mask = Image.open(mask_path)
        mask = mask.resize(self.img_size, Image.NEAREST)

        mask_np = np.array(mask)
        binary_mask = np.zeros_like(mask_np, dtype=np.float32)
        # 1: pet -> foreground, 2/3 -> background
        binary_mask[mask_np == 1] = 1.0

        mask = torch.from_numpy(binary_mask).unsqueeze(0)  # [1, H, W]
        return mask

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img_path = os.path.join(self.img_dir, file_name + '.jpg')

        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.img_size, Image.BILINEAR)
        image = TF.to_tensor(image)

        if self.has_mask:
            mask = self._load_mask(file_name)
        else:
            # dummy mask for inference
            mask = torch.zeros((1, self.img_size[1], self.img_size[0]), dtype=torch.float32)

        # Basic Data Augmentation
        if self.is_train:
            # 1. horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 2. affine augmentation
            if random.random() > 0.5:
                angle = random.uniform(self.aug_angle_range[0], self.aug_angle_range[1])
                max_dx = int(0.1 * self.img_size[0])
                max_dy = int(0.1 * self.img_size[1])
                translate = (random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy))
                scale = random.uniform(self.aug_scale_range[0], self.aug_scale_range[1])

                image = TF.affine(
                    image,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=0,
                    interpolation=InterpolationMode.BILINEAR,
                )
                mask = TF.affine(
                    mask,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=0,
                    interpolation=InterpolationMode.NEAREST,
                )

            # 3. color jitter-like perturbation (image only)
            if random.random() > 0.5:
                brightness = random.uniform(0.8, 1.2)
                contrast = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness)
                image = TF.adjust_contrast(image, contrast)

        # Normalize image using ImageNet stats
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Reflect padding only for original UNet input preprocessing
        if self.pad_for_unet:
            p = self.unet_pad
            image = F.pad(image, (p, p, p, p), mode='reflect')
            # mask stays 320x320, because model output will return to 320x320

        return image, mask, file_name