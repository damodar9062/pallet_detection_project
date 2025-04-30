import os
import cv2
import numpy as np

def convert_txt_to_png(txt_path, image_path, output_mask_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x_center *= w
            y_center *= h
            width *= w
            height *= h
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)

    cv2.imwrite(output_mask_path, mask)

def main():
    txt_dir = '/home/dhamodarlinux/ros2_ws/data/pallets/pallet_detection/test/labels'
    image_dir = '/home/dhamodarlinux/ros2_ws/data/pallets/pallet_detection/test/images'
    output_dir = '/home/dhamodarlinux/ros2_ws/data/pallets/pallet_detection/test/masks'
    os.makedirs(output_dir, exist_ok=True)

    for txt_file in os.listdir(txt_dir):
        if txt_file.endswith('.txt'):
            txt_path = os.path.join(txt_dir, txt_file)
            image_name = txt_file.replace('.txt', '.jpg')
            image_path = os.path.join(image_dir, image_name)
            output_mask_path = os.path.join(output_dir, txt_file.replace('.txt', '.png'))
            if os.path.exists(image_path):
                convert_txt_to_png(txt_path, image_path, output_mask_path)

if __name__ == '__main__':
    main()