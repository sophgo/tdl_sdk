import sys
import os
from tdl import nn,image

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_fd.py <model_path> <image_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    classifier = nn.Classifier(nn.ModelType.CLS_RGBLIVENESS, model_path)
    img = image.read(img_path)
    classification = classifier.inference(img)
    print(classification)
    