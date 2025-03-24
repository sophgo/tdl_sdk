import sys
import os
from tdl import nn,image

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_face_attribute.py <model_path> <image_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    face_attribute = nn.AttributeExtractor(nn.ModelType.CLS_ATTRIBUTE_FACE, model_path)
    img = image.read(img_path)
    face_attribution = face_attribute.inference(img)
    print(face_attribution)