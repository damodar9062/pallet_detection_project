import json
import os

def filter_coco_json(coco_json_path, image_dir, output_json_path):
    with open(coco_json_path, 'r') as f:
        data = json.load(f)
    image_files = set(f for f in os.listdir(image_dir) if f.endswith('.jpg'))
    filtered_images = [img for img in data['images'] if img['file_name'] in image_files]
    filtered_image_ids = set(img['id'] for img in filtered_images)
    filtered_annotations = [ann for ann in data['annotations'] if ann['image_id'] in filtered_image_ids]
    data['images'] = filtered_images
    data['annotations'] = filtered_annotations
    with open(output_json_path, 'w') as f:
        json.dump(data, f)
    print(f"Filtered {coco_json_path}: {len(filtered_images)} images, {len(filtered_annotations)} annotations")

base_dir = 'ground_segmentation'
for split in ['train', 'valid', 'test']:
    coco_json = f'{base_dir}/{split}/_annotations.coco.json'
    image_dir = f'{base_dir}/{split}'
    output_json = f'{base_dir}/{split}/filtered_annotations.coco.json'
    if os.path.exists(coco_json):
        filter_coco_json(coco_json, image_dir, output_json)
        os.replace(output_json, coco_json)
    else:
        print(f"Warning: {coco_json} not found")