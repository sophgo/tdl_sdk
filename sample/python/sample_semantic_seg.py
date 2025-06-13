import sys
import os
from tdl import nn,image

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_semantic_seg.py <model_path> <image_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    semantic_segmentator = nn.get_model(nn.ModelType.TOPFORMER_SEG_PERSON_FACE_VEHICLE, model_path)
    img = image.read(img_path)
    semantic_segmentation = semantic_segmentator.inference(img)
    # 获取列表中的第一个元素
    result = semantic_segmentation[0]
    print(result)
    for y in range(result["output_height"]):
        for x in range(result["output_width"]):
            index = y * result["output_width"] + x
            print(result["class_id"][index], end=' ')
        print() 
    