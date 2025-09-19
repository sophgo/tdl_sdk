import sys
import os
from tdl import nn, image
import cv2
import numpy as np

SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]

def crop_expand(img, bbox, expansion=1.25):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    new_width = int(width * expansion)
    new_height = int(height * expansion)
    crop_x1 = max(x1 - (new_width - width) // 2, 0)
    crop_y1 = max(y1 - (new_height - height) // 2, 0)
    crop_x2 = min(crop_x1 + new_width, img.shape[1])
    crop_y2 = min(crop_y1 + new_height, img.shape[0])
    crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    return crop_img, (crop_x1, crop_y1)

def visualize_keypoints(img_path, persons, save_path="keypoint_detection.jpg", score_thr=0.5):
    img = cv2.imread(img_path)
    if not isinstance(persons, list):
        print("输出格式错误")
        return
    print(f"检测到 {len(persons)} 个人体")
    for idx, person in enumerate(persons):
        if isinstance(person, dict) and "landmarks" in person and isinstance(person["landmarks_score"], list):
            kpts = []
            for i in range(len(person["landmarks"])):
                kpts.append({
                    "x": person["landmarks"][i][0],
                    "y": person["landmarks"][i][1],
                    "score": person["landmarks_score"][i]
                })
        elif isinstance(person, dict) and "class_id" in person:
            kpts = []
            for i in range(len(person["landmarks"])):
                kpts.append({
                    "x": person["landmarks"][i][0],
                    "y": person["landmarks"][i][1],
                    "score": person.get("score", 1)
                })
        elif isinstance(person, dict) and "keypoints" in person:
            kpts = person["keypoints"]
        elif isinstance(person, dict) and {"x", "y", "score"}.issubset(person.keys()):
            kpts = [person]
        elif isinstance(person, list):
            kpts = person
        else:
            print("未知关键点格式", person)
            continue

        for i, kp in enumerate(kpts):
            if kp.get("score", 1) < score_thr:
                continue
            x, y = int(kp["x"]), int(kp["y"])
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(img, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for i, (a, b) in enumerate(SKELETON):
            if a >= len(kpts) or b >= len(kpts):
                continue
            if kpts[a].get("score", 1) < score_thr or kpts[b].get("score", 1) < score_thr:
                continue
            pt1 = (int(kpts[a]["x"]), int(kpts[a]["y"]))
            pt2 = (int(kpts[b]["x"]), int(kpts[b]["y"]))
            cv2.line(img, pt1, pt2, (255, 128, 0), 2)
    cv2.imwrite(save_path, img)
    print(f"保存图像到: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python sample_img_human_keypoint.py <mode:simcc|yolov8> <model_dir> <image_path>")
        sys.exit(1)

    mode = sys.argv[1]
    model_dir = sys.argv[2]
    image_path = sys.argv[3]

    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        sys.exit(1)

    if mode == "simcc":
        # 1. 人体检测
        det_model = nn.get_model_from_dir(nn.ModelType.MBV2_DET_PERSON, model_dir)
        img = image.read(image_path)
        det_results = det_model.inference(img)
        # 2. 裁剪每个人体
        img_cv = cv2.imread(image_path)
        human_crops = []
        crop_boxes = []
        for i, person in enumerate(det_results):
            # 假设person有x1,y1,x2,y2字段
            bbox = [int(person["x1"]), int(person["y1"]), int(person["x2"]), int(person["y2"])]
            crop_img, (crop_x1, crop_y1) = crop_expand(img_cv, bbox, expansion=1.25)
            crop_boxes.append((crop_x1, crop_y1))
            crop_path = f"human_crop_{i}.jpg"
            cv2.imwrite(crop_path, crop_img)
            human_crops.append(crop_img)
        # 3. 关键点推理
        keypoint_model = nn.get_model_from_dir(nn.ModelType.KEYPOINT_SIMCC_PERSON17, model_dir)
        persons = []
        for i, crop_img in enumerate(human_crops):
            tmp_crop_path = f"tmp_crop_{i}.jpg"
            cv2.imwrite(tmp_crop_path, crop_img)
            crop_img = image.read(tmp_crop_path)
            kp_result = keypoint_model.inference(crop_img)
            if isinstance(kp_result, list):
                if len(kp_result) == 0:
                    continue
                kp_result = kp_result[0]
            for j in range(len(kp_result["landmarks"])):
                kp_result["landmarks"][j][0] += crop_boxes[i][0]
                kp_result["landmarks"][j][1] += crop_boxes[i][1]
            persons.append(kp_result)
        visualize_keypoints(image_path, persons, save_path="simcc_keypoints.jpg", score_thr=3)
    elif mode == "yolov8":
        model = nn.get_model_from_dir(nn.ModelType.KEYPOINT_YOLOV8POSE_PERSON17, model_dir)
        img = image.read(image_path)
        outdatas = model.inference(img)
        if isinstance(outdatas, dict):
            outdatas = [outdatas]
        elif isinstance(outdatas, list) and isinstance(outdatas[0], dict):
            pass
        else:
            print("yolov8输出格式错误", outdatas)
            sys.exit(1)
        visualize_keypoints(image_path, outdatas, save_path="yolov8_keypoints.jpg", score_thr=0.5)
    else:
        print("Unknown mode, use simcc or yolov8")
        sys.exit(1)

# input: sample_img_human_keypoint.py <mode:simcc|yolov8> <model_dir> <image_path>
# output: <smicc_keypoints.jpg|yolov8_keypoints.jpg>