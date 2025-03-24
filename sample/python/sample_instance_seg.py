import sys
import os
from tdl import nn,image

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_instance_seg.py <model_path> <image_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    instance_segmentator = nn.InstanceSegmentation(nn.ModelType.YOLOV8_SEG_COCO80, model_path)
    img = image.read(img_path)
    instance_segmentation = instance_segmentator.inference(img)
    print(instance_segmentation)

    