import sys
import cv2
import tempfile
import os
import numpy as np
from tdl import nn, image


def draw_license_plate(img, bbox, landmarks, text=None):
    x1, y1, x2, y2 = map(int, [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for (x, y) in landmarks:
        cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
    if text:
        cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

def align_license_plate(img, landmarks):
    dst_pts = np.array([[0,0], [136,0], [136,36], [0,36]], dtype=np.float32)
    src_pts = np.array(landmarks, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    aligned = cv2.warpPerspective(img, M, (136,36))
    return aligned

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 sample_image_license_plate_recognition.py <model_dir> <image_path>")
        sys.exit(1)
    model_dir = sys.argv[1]
    img_path = sys.argv[2]

    img = cv2.imread(img_path)
    tdl_img = image.read(img_path)


    det_model = nn.get_model_from_dir(nn.ModelType.YOLOV8N_DET_LICENSE_PLATE, model_dir, device_id=0)
    kpt_model = nn.get_model_from_dir(nn.ModelType.KEYPOINT_LICENSE_PLATE, model_dir, device_id=0)
    rec_model = nn.get_model_from_dir(nn.ModelType.RECOGNITION_LICENSE_PLATE, model_dir, device_id=0)

    det_results = det_model.inference(tdl_img)
    for i, plate in enumerate(det_results):
        bbox = plate

        crop_img = img[int(bbox['y1']):int(bbox['y2']), int(bbox['x1']):int(bbox['x2'])]

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpfile:
            temp_crop_path = tmpfile.name
            cv2.imwrite(temp_crop_path, crop_img)
        tdl_crop = image.read(temp_crop_path)
        os.remove(temp_crop_path)
        kpt_results = kpt_model.inference(tdl_crop)

        if 'landmarks' in kpt_results[0]:
            landmarks = kpt_results[0]['landmarks']
        else:
            landmarks = list(zip(kpt_results[0]['landmarks_x'], kpt_results[0]['landmarks_y']))

        aligned_img = align_license_plate(crop_img, landmarks)
        cv2.imwrite(f"aligned_plate_{i}.jpg", aligned_img)

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpfile:
            temp_aligned_path = tmpfile.name
            cv2.imwrite(temp_aligned_path, aligned_img)

        tdl_aligned = image.read(temp_aligned_path)
        os.remove(temp_aligned_path)
        rec_results = rec_model.inference(tdl_aligned)
        plate_text = rec_results[0]['text'] if 'text' in rec_results[0] else str(rec_results[0])

        draw_license_plate(img, bbox, landmarks, plate_text)
        print("keypoints:")
        for i, (x, y) in enumerate(landmarks):
            print(f"{i}: {x:.2f} {y:.2f}")
        print(f"Plate {i}: {plate_text}")
    cv2.imwrite("result_license_plate.jpg", img)
    print("可视化结果已保存为: result_license_plate.jpg")