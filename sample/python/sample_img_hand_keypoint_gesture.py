import sys
import os
import cv2
from tdl import nn, image
import numpy as np

def crop_and_expand(img, bbox, expansion=1.25):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    new_width = int(width * expansion)
    new_height = int(height * expansion)
    crop_x1 = int(max(x1 - (new_width - width) // 2, 0))
    crop_y1 = int(max(y1 - (new_height - height) // 2, 0))
    crop_x2 = int(min(crop_x1 + new_width, img.shape[1]))
    crop_y2 = int(min(crop_y1 + new_height, img.shape[0]))
    crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    return crop_img, (crop_x1, crop_y1)


def visualize_keypoints(img, landmarks, save_path):
    for x, y in landmarks:
        cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)
    print(f"保存图像到: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 sample_img_hand_keypoint_gesture.py <model_dir> <image_path>")
        sys.exit(1)

    model_dir = sys.argv[1]
    image_path = sys.argv[2]

    model_hd = nn.get_model_from_dir(nn.ModelType.YOLOV8N_DET_HAND, model_dir)
    model_hk = nn.get_model_from_dir(nn.ModelType.KEYPOINT_HAND, model_dir)
    model_hc = nn.get_model_from_dir(nn.ModelType.CLS_KEYPOINT_HAND_GESTURE, model_dir)

    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        sys.exit(1)
    
    img = image.read(image_path)
    img_cv = cv2.imread(image_path)

    out_hd = model_hd.inference(img)
    if len(out_hd) == 0:
        print("No hand detected")
        sys.exit(0)

    hand_crops = []
    crop_offsets = []
    for i, box in enumerate(out_hd):
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        crop_img, (off_x, off_y) = crop_and_expand(img_cv, (x1, y1, x2, y2), expansion=1.25)
        crop_offsets.append((off_x, off_y))
        crop_img_path = f"hand_crop_{i}.jpg"
        cv2.imwrite(crop_img_path, crop_img)
        hand_crops.append(image.read(crop_img_path))

    out_hk = []
    for crop in hand_crops:
        out_hk.extend(model_hk.inference(crop))


    out_hc = []
    for hk in out_hk:
        keypoints = []
        for pt in hk["landmarks"]:
            keypoints.extend(pt)
        # 关键点输入为 42x1x1 float32 单通道
        bin_data = image.Image.from_numpy(
            np.array(keypoints, dtype="float32").reshape(42, 1, 1),
            image.ImageFormat.GRAY
        )
        out_hc.extend(model_hc.inference(bin_data))

    for i, (hk, hc) in enumerate(zip(out_hk, out_hc)):
        landmarks = []
        for idx, pt in enumerate(hk["landmarks"]):
            x = pt[0] * hand_crops[i].get_size()[0] + crop_offsets[i][0]
            y = pt[1] * hand_crops[i].get_size()[1] + crop_offsets[i][1]
            landmarks.append((x, y))
            print(f"{idx}: {x:.2f} {y:.2f}")
        print(f"hand[{i}]: label: {hc['class_id']}, score: {hc['score']:.2f}")
        save_path = f"hand_keypoints_{i}_label_{hc['class_id']}.jpg"
        visualize_keypoints(img_cv.copy(), landmarks, save_path)

# input: python3 sample_img_hand_keypoint_gesture.py <model_dir> <image_path>
# output: hand[i]: label: <class_id>, score: <score>