import sys
import os
from tdl import nn, image
import cv2

def crop_and_expand(img, bbox, expansion=1.125):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    new_width = int(width * expansion)
    new_height = int(height * expansion)
    crop_x1 = int(x1 - (new_width - width) / 2)
    crop_y1 = int(y1 - (new_height - height) / 2)
    # 边界检查
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(crop_x1 + new_width, img.shape[1])
    crop_y2 = min(crop_y1 + new_height, img.shape[0])
    w = crop_x2 - crop_x1
    h = crop_y2 - crop_y1
    return img[crop_y1:crop_y1+h, crop_x1:crop_x1+w]

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 sample_img_hand_gesture.py <model_path1> <model_path2> <image_path>")
        sys.exit(1)

    model_path1 = sys.argv[1]
    model_path2 = sys.argv[2]
    image_path = sys.argv[3]

    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        sys.exit(1)

    model_config = {
        "mean": (2.1179, 2.0357, 1.8044),
        "scale": (0.017126, 0.017509, 0.017431)
    }

    model_hd = nn.get_model(nn.ModelType.YOLOV8N_DET_HAND, model_path1)
    model_hc = nn.get_model(nn.ModelType.CLS_HAND_GESTURE, model_path2, model_config)

    img = image.read(image_path)
    img_cv = cv2.imread(image_path)

    out_datas = model_hd.inference(img)
    if len(out_datas) == 0:
        print("No object detected")
        sys.exit(0)
    
    hand_crops = []
    for j, obj in enumerate(out_datas):
        bbox = (obj['x1'], obj['y1'], obj['x2'], obj['y2'])
        print(f"hand_bbox_{j}: {bbox}")
        cropped_img = crop_and_expand(img_cv, bbox)
        cropped_img_resized = cv2.resize(cropped_img, (128, 128))
        cv2.imwrite(f"debug_hand_crop_{j}.jpg", cropped_img_resized)
        hand_crops.append(image.Image.from_numpy(cropped_img_resized, image.ImageFormat.BGR_PACKED))
        
    results = []
    for i, hand_img in enumerate(hand_crops):
        out_cls = model_hc.inference(hand_img)
        print(f"hand_crop_{i} out_cls: {out_cls}")
        if out_cls and isinstance(out_cls[0], dict):
            label = out_cls[0].get('topk_class_ids', [out_cls[0].get('class_id', -1)])[0]
            score = out_cls[0].get('topk_scores', [out_cls[0].get('score', 0.0)])[0]
            print(f"hand_crop_{i}, label: {label}, score: {score:.2f}")

            print(f"保存裁剪图像到: hand_crop_{i}_label_{label}.jpg")
        else:
            print(f"hand_crop_{i}, 无法识别手势")

# input: python3 sample_img_hand_gesture.py <model_path1> <model_path2> <image_path>
# output: hand_crop_{i}, label: {label}, score: {score:.2f}