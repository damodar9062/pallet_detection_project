import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights
import albumentations as A
import numpy as np

class PalletDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = self.images[idx]
        img_path = os.path.join(self.image_dir, img_file)
        mask_path = os.path.join(self.mask_dir, img_file.replace('.jpg', '.png'))

        # Load image and mask
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Normalize mask to {0, 1}
        mask = (mask > 0).astype(np.uint8)  # Convert 255 to 1

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        # Convert to tensors
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # HWC to CHW, normalize
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

# Define transformations
transform = A.Compose([
    A.Resize(520, 520),  # DeepLabV3 expects 520x520 input
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ImageNet normalization
])

# Create datasets
train_dataset = PalletDataset(
    '/home/dhamodarlinux/ros2_ws/data/pallets/ground_segmentation/train',
    '/home/dhamodarlinux/ros2_ws/data/pallets/ground_segmentation/train/masks',
    transform=transform
)
valid_dataset = PalletDataset(
    '/home/dhamodarlinux/ros2_ws/data/pallets/ground_segmentation/valid',
    '/home/dhamodarlinux/ros2_ws/data/pallets/ground_segmentation/valid/masks',
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4)

# Initialize model
model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)  # 2 classes: background, ground
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')

# Save model
torch.save(model.state_dict(), '/home/dhamodarlinux/ros2_ws/data/pallets/deeplabv3.pth')

# Evaluation
def calculate_iou(pred, target):
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    return intersection / union if union > 0 else 0

model.eval()
iou_scores = []
with torch.no_grad():
    for images, masks in valid_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)['out']
        preds = torch.argmax(outputs, dim=1)
        for pred, mask in zip(preds, masks):
            pred = pred.cpu().numpy()
            mask = mask.cpu().numpy()
            iou_scores.append(calculate_iou(pred == 1, mask == 1))
mean_iou = np.mean(iou_scores)
print(f'Validation Mean IoU: {mean_iou:.4f}')

# Save evaluation results
with open('/home/dhamodarlinux/ros2_ws/data/pallets/evaluation.txt', 'a') as f:
    f.write(f"Ground Segmentation Validation IoU: {mean_iou}\n")