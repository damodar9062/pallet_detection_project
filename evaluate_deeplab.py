
import torch
from torchvision.models.segmentation import deeplabv3_resnet50
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os

class PalletDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))  # Adjust extension
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        mask = torch.tensor(mask).long()
        return image, mask

def calculate_iou(pred, target, num_classes=2):
    iou = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            iou.append(float('nan'))
        else:
            iou.append(intersection / union)
    return np.nanmean(iou)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = deeplabv3_resnet50(weights=None, num_classes=2, aux_loss=False)
    model.load_state_dict(torch.load('deeplabv3.pth', map_location='cpu', weights_only=True), strict=False)
    model.to(device)
    model.eval()

    transform = A.Compose([
        A.Resize(520, 520),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    test_dataset = PalletDataset(
        image_dir='/home/dhamodarlinux/ros2_ws/data/pallets/pallet_detection/test/images',
        mask_dir='/home/dhamodarlinux/ros2_ws/data/pallets/pallet_detection/test/labels',  # Adjust path
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    ious = []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)
            iou = calculate_iou(preds.cpu().numpy(), masks.cpu().numpy())
            ious.append(iou)

    mean_iou = np.nanmean(ious)
    print(f'Test IoU: {mean_iou:.4f}')

if __name__ == '__main__':
    main()