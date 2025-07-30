import sys
import os
from tdl import nn,image
import cv2

def visualize_lane_detection(img_path, box_landmarks, save_path="lstr_lane_detection.jpg"):
    img = cv2.imread(img_path)
    for lane in box_landmarks:
        pt1 = tuple(map(int, lane['landmarks'][0]))
        pt2 = tuple(map(int, lane['landmarks'][1]))
        cv2.line(img, pt1, pt2, (0, 255, 0), 3)
    cv2.imwrite(save_path, img)
    print(f"保存图像到: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_lane_detection.py <model_dir> <image_path>")
        sys.exit(1)
    model_dir = sys.argv[1]
    img_path = sys.argv[2]
    if not os.path.exists(img_path):
        print(f"图像文件不存在: {img_path}")
        sys.exit(1)
    lane_detector = nn.get_model_from_dir(nn.ModelType.LSTR_DET_LANE, model_dir)
    img = image.read(img_path)
    out_datas = lane_detector.inference(img)
    
    if len(out_datas) == 0:
        print("No object detected")
    else:
        for j, lane in enumerate(out_datas):
            print(f"lane {j}:")
            for k, (x, y) in enumerate(lane['landmarks']):
                print(f"{k}: {x}  {y}")
        
        visualize_lane_detection(img_path, out_datas, "lstr_lane_detection.jpg")

# input: python sample_img_lane_detection.py <model_dir> <image_path>
# output: lane i: 0: <x1> <y1> 1: <x2> <y2>
    