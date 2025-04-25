import cv2
import json
import numpy as np
import os

def convert_coco_to_mask(coco_json, image_dir, output_mask_dir):
    with open(coco_json, 'r') as f:
        data = json.load(f)
    os.makedirs(output_mask_dir, exist_ok=True)
    image_files = set(f for f in os.listdir(image_dir) if f.endswith('.jpg'))
    for img in data['images']:
        img_id = img['id']
        img_name = img['file_name']
        if img_name not in image_files:
            print(f"Warning: Image {img_name} not found in {image_dir}")
            continue
        img_path = os.path.join(image_dir, img_name)
        img_data = cv2.imread(img_path)
        if img_data is None:
            print(f"Error: Failed to load image {img_path}")
            continue
        # Use actual image dimensions
        height, width = img_data.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        annotations_found = False
        for ann in data['annotations']:
            if ann['image_id'] == img_id and ann['category_id'] == 1:  # Ground class
                annotations_found = True
                for seg in ann['segmentation']:
                    points = np.array(seg).reshape(-1, 2).astype(np.int32)
                    # Clip coordinates to image bounds
                    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
                    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
                    cv2.fillPoly(mask, [points], 1)
        if not annotations_found:
            print(f"Warning: No annotations for {img_name} with category_id=1")
        mask_path = os.path.join(output_mask_dir, img_name.replace('.jpg', '.png'))
        cv2.imwrite(mask_path, mask * 255)  # Scale to 0-255 for visibility
        print(f"Generated mask: {mask_path} (non-zero pixels: {np.sum(mask)})")
        # Save a debug image overlaying mask on original
        debug_img = img_data.copy()
        debug_img[mask == 1] = [0, 255, 0]  # Green for ground
        debug_path = os.path.join(output_mask_dir, f"debug_{img_name}")
        cv2.imwrite(debug_path, debug_img)

# Convert for each split
base_dir = 'ground_segmentation'
for split in ['train', 'valid', 'test']:
    image_dir = f'{base_dir}/{split}'
    output_mask_dir = f'{base_dir}/{split}/masks'
    coco_json = f'{base_dir}/{split}/_annotations.coco.json'
    if not os.path.exists(coco_json):
        print(f"Error: {coco_json} not found")
        continue
    if not os.path.exists(image_dir):
        print(f"Error: Image directory {image_dir} not found")
        continue
    convert_coco_to_mask(coco_json, image_dir, output_mask_dir)