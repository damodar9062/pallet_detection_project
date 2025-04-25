import os
import cv2
import albumentations as A
import numpy as np

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.GaussianBlur(p=0.3),
])

def augment_images(image_dir, label_dir, mask_dir, output_dir, num_augs=2):
    os.makedirs(f'{output_dir}/images', exist_ok=True)
    if label_dir:
        os.makedirs(f'{output_dir}/labels', exist_ok=True)
    if mask_dir:
        os.makedirs(f'{output_dir}/masks', exist_ok=True)
    for img_file in os.listdir(image_dir):
        if img_file.endswith('.jpg'):
            img = cv2.imread(os.path.join(image_dir, img_file))
            label_file = img_file.replace('.jpg', '.txt') if label_dir else None
            mask_file = img_file.replace('.jpg', '.png') if mask_dir else None
            mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE) if mask_dir and os.path.exists(os.path.join(mask_dir, mask_file)) else None
            labels = []
            if label_file and os.path.exists(os.path.join(label_dir, label_file)):
                with open(os.path.join(label_dir, label_file), 'r') as f:
                    labels = f.readlines()
            for i in range(num_augs):
                aug = transform(image=img, mask=mask) if mask_dir else transform(image=img)
                aug_img, aug_mask = aug['image'], aug.get('mask')
                aug_img_name = f'aug_{i}_{img_file}'
                cv2.imwrite(f'{output_dir}/images/{aug_img_name}', aug_img)
                if aug_mask is not None:
                    cv2.imwrite(f'{output_dir}/masks/aug_{i}_{mask_file}', aug_mask)
                if labels:
                    with open(f'{output_dir}/labels/aug_{i}_{label_file}', 'w') as f:
                        f.writelines(labels)  # Simplified for detection

# Augment detection dataset
augment_images('pallet_detection/train/images',
               'pallet_detection/train/labels',
               None,
               'pallet_detection/train/images')
# Augment segmentation dataset
augment_images('ground_segmentation/train',
               None,
               'ground_segmentation/train/masks',
               'ground_segmentation/train')