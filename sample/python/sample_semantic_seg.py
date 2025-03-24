import sys
import os
from tdl import nn,image

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_semantic_seg.py <model_path> <image_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    semantic_segmentator = nn.SemanticSegmentation(nn.ModelType.TOPFORMER_SEG_PERSON_FACE_VEHICLE, model_path)
    img = image.read(img_path)
    semantic_segmentation = semantic_segmentator.inference(img)
    print(semantic_segmentation)
    for y in range(semantic_segmentation["output_height"]):
        for x in range(semantic_segmentation["output_width"]):
            index = y * semantic_segmentation["output_width"] + x
            print(semantic_segmentation["class_id"][index], end=' ')
        print() 
    