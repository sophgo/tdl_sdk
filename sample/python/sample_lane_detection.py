import sys
import os
from tdl import nn,image

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_lane_detection.py <model_path> <image_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    lane_detector = nn.get_model(nn.ModelType.LSTR_DET_LANE, model_path)
    img = image.read(img_path)
    detection = lane_detector.inference(img)
    print(detection)
    