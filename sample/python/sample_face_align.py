import sys
import os
from tdl import nn, image
import cv2
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_face_align.py <face_model_path> <image_path>")
        sys.exit(1)
    
    face_model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    # 读取图像
    img = cv2.imread(image_path)
    img_tdl = image.Image.from_numpy(img)
    
    # 创建人脸检测器并进行检测
    face_detector = nn.FaceDetector(nn.ModelType.SCRFD_DET_FACE, face_model_path)
    print("正在执行人脸检测...")
    bboxes = face_detector.inference(img)
    
    if not bboxes:
        print("未检测到人脸!")
        sys.exit(1)
    
    print(f"检测到 {len(bboxes)} 个人脸")
    
    # 处理第一个检测到的人脸的关键点,如果有多个人脸，循环len(bboxes)
    face_info = bboxes[0]
    landmarks = face_info['landmarks']
    #print(landmarks)
    
    # 将关键点转换为一维向量格式 [x1,y1,x2,y2,...]
    src_landmarks = []
    for landmark in landmarks:
        src_landmarks.extend([float(landmark[0]), float(landmark[1])])
    
    # 定义目标关键点（标准5点模板）
    dst_landmarks = [
        38.2946, 51.6963,  # 左眼
        73.5318, 51.5014,  # 右眼
        56.0252, 71.7366,  # 鼻子
        41.5493, 92.3655,  # 左嘴角
        70.7299, 99.3655   # 右嘴角
    ]

    print("正在执行人脸对齐...")
    # 调用人脸对齐函数
    aligned_img = image.align_face(img_tdl, src_landmarks, dst_landmarks, 5)
    
    # 保存结果
    output_path = "aligned_face.jpg"
    image.write(aligned_img, output_path)
    print(f"对齐后的人脸已保存至: {output_path}")
