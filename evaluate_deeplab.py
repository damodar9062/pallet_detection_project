import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import albumentations as A
import numpy as np


def calculate_iou(pred, target, num_classes=2):
    """
    Compute mean Intersection-over-Union (IoU) over all classes.
    """
    iou_list = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            continue
        iou_list.append(intersection / union)
    if not iou_list:
        return float('nan')
    return float(np.mean(iou_list))


class PalletDataset(Dataset):
    """
    PyTorch Dataset for loading image/mask pairs for segmentation.
    Expects images in one directory (e.g. .jpg) and masks in another (.png),
    with matching base filenames.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image path
        img_filename = self.images[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        # Derive mask filename by replacing extension
        base = os.path.splitext(img_filename)[0]
        mask_filename = base + '.png'
        mask_path = os.path.join(self.mask_dir, mask_filename)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask


def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = deeplabv3_resnet50(weights=None, num_classes=2, aux_loss=False)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    model.to(device)
    model.eval()

    # Load trained weights
    state_dict = torch.load('deeplabv3.pth', map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # Define preprocessing transforms (must match training)
    transform = A.Compose([
        A.Resize(520, 520),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Create test dataset & loader
    test_dataset = PalletDataset(
        image_dir='/home/dhamodarlinux/ros2_ws/data/pallets/ground_segmentation/test',
        mask_dir='/home/dhamodarlinux/ros2_ws/data/pallets/ground_segmentation/test/masks',
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluate IoU
    ious = []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)
            # Compute IoU per sample
            iou = calculate_iou(preds.cpu().numpy(), masks.cpu().numpy(), num_classes=2)
            ious.append(iou)

    mean_iou = float(np.nanmean(ious))
    print(f'Test Mean IoU: {mean_iou:.4f}')


if __name__ == '__main__':
    main()
