import sys
import os
from tdl import nn,image

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_feature_extraction.py <model_path> <image_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    feature_extractor = nn.FeatureExtractor(nn.ModelType.CLIP_FEATURE_IMG, model_path)
    img = image.read(img_path)
    feature = feature_extractor.inference(img)
    print(feature)

    