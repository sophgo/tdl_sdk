import sys
import os
import cv2
import numpy as np
from tdl import nn,image

def visualize_mask(seg_result, img_name):
    """
    可视化语义分割掩码
    Args:
        seg_result: 分割结果字典，包含class_id、output_width、output_height等
        img_name: 保存的图像文件名
    """
    output_width = seg_result["output_width"]
    output_height = seg_result["output_height"]
    class_id = seg_result["class_id"]
    
    mask_array = np.array(class_id, dtype=np.uint8).reshape(output_height, output_width)
    

    dst = mask_array * 50

    cv2.imwrite(img_name, dst)
    print(f"掩码图像已保存到: {img_name}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_semantic_seg.py <model_dir> <image_path>")
        sys.exit(1)
    model_dir = sys.argv[1]
    img_path = sys.argv[2]
    semantic_segmentator = nn.get_model_from_dir(nn.ModelType.TOPFORMER_SEG_PERSON_FACE_VEHICLE, model_dir)
    img = image.read(img_path)
    semantic_segmentation = semantic_segmentator.inference(img)

    result = semantic_segmentation[0]
    print(result)
    
    visualize_mask(result, "topfoemer_seg_mask.png")
    
    for y in range(result["output_height"]):
        for x in range(result["output_width"]):
            index = y * result["output_width"] + x
            print(result["class_id"][index], end=' ')
        print()