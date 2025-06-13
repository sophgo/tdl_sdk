import sys
import os
from tdl import nn,image

PreprocessParameters = {
    "mean": (0, 0, 0),
    "scale": (255.0,255.0,255.0),
}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_fd.py <model_path> <image_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    classifier = nn.get_model(nn.ModelType.CLS_RGBLIVENESS, model_path, PreprocessParameters)
    img = image.read(img_path)
    classification = classifier.inference(img)
    # 获取列表中的第一个字典
    result = classification[0]
    print(result["class_id"])
    print(result["score"])
    