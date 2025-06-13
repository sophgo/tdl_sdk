import sys
import os
from tdl import nn,image

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_keypoints.py <model_path> <image_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    keypoint_detector = nn.get_model(nn.ModelType.KEYPOINT_LICENSE_PLATE, model_path)
    img = image.read(img_path)
    keypoints = keypoint_detector.inference(img)
    result = keypoints[0]
    print(result)

    