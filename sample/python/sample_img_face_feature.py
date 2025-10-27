import sys
import os
from tdl import nn, image
import cv2
import numpy as np
import math

def face_crop_align(face_crop, landmarks):
    """人脸对齐函数"""
    # 将关键点转换为一维向量格式 [x1,y1,x2,y2,...]
    landmarks_flat = []
    for i in range(5):
        landmarks_flat.extend([float(landmarks[i][0]), float(landmarks[i][1])])
    
    # 定义目标关键点（标准5点模板）
    dst_landmarks = [
        38.2946, 51.6963,  # 左眼
        73.5318, 51.5014,  # 右眼
        56.0252, 71.7366,  # 鼻子
        41.5493, 92.3655,  # 左嘴角
        70.7299, 92.3655   # 右嘴角
    ]
    
    aligned_face = image.align_face(face_crop, landmarks_flat, dst_landmarks, 5)
    return aligned_face

def extract_crop_face_landmark(model_fl, images, face_metas):
    """提取人脸裁剪和关键点"""
    face_crops_landmark = []
    face_crops = []
    
    for i, img in enumerate(images):
        # 获取第一个人脸的边界框
        if len(face_metas[i]) == 0:
            continue
            
        face_info = face_metas[i][0]
        x1, y1, x2, y2 = int(face_info['x1']), int(face_info['y1']), int(face_info['x2']), int(face_info['y2'])
        img_width, img_height = img.get_size()
        
        # 扩大1.2倍裁剪区域
        box_width = x2 - x1
        box_height = y2 - y1
        crop_size = int(max(box_width, box_height) * 1.2)
        crop_x1 = x1 - (crop_size - box_width) // 2
        crop_y1 = y1 - (crop_size - box_height) // 2
        crop_x2 = x2 + (crop_size - box_width) // 2
        crop_y2 = y2 + (crop_size - box_height) // 2
        
        # 边界检查
        crop_x1 = max(crop_x1, 0)
        crop_y1 = max(crop_y1, 0)
        crop_x2 = min(crop_x2, img_width)
        crop_y2 = min(crop_y2, img_height)

        # 裁剪人脸
        roi = (crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1)
        face_crop = image.crop(img, roi)
        print(f"crop_x1: {crop_x1} crop_y1: {crop_y1} crop_x2: {crop_x2} crop_y2: {crop_y2}")
        
        # 保存裁剪的人脸
        image.write(face_crop, f"face_crop_{i}.jpg")
        face_crops.append(face_crop)
    
    # 对裁剪的人脸进行关键点检测
    for i, face_crop in enumerate(face_crops):
        landmarks_result = model_fl.inference(face_crop)
        if len(landmarks_result) > 0:
            face_crops_landmark.append((face_crop, landmarks_result[0]))
    
    return face_crops_landmark

def visualize_face_crop(face_crops_landmark):
    """可视化人脸裁剪和关键点"""
    for i, (face_crop, landmarks_meta) in enumerate(face_crops_landmark):
        temp_path = f"temp_face_{i}.jpg"
        try:
            image.write(face_crop, temp_path)
            face_np = cv2.imread(temp_path)
            
            # 绘制关键点
            for landmark in landmarks_meta['landmarks']:
                cv2.circle(face_np, (int(landmark[0]), int(landmark[1])), 2, (0, 0, 255), -1)
            
            # 保存标注了关键点的图像
            cv2.imwrite(f"face_crop_landmark_{i}.jpg", face_np)
            
        except Exception as e:
            print(f"Error processing face {i}: {e}")
        finally:
            # 确保删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python sample_img_face_feature.py <feature_extraction_model_id_name> <model_dir> <image_path1> <image_path2>")
        print("feature_extraction_model_id_name:")
        print("FEATURE_BMFACE_R34")
        print("FEATURE_BMFACE_R50")
        print("FEATURE_CVIFACE")
        sys.exit(1)

    model_id_name = sys.argv[1]
    model_dir = sys.argv[2]
    image_path1 = sys.argv[3]
    image_path2 = sys.argv[4]

    # 检查图像文件是否存在
    if not os.path.exists(image_path1):
        print(f"Failed to create image1")
        sys.exit(1)
    if not os.path.exists(image_path2):
        print(f"Failed to create image2")
        sys.exit(1)

    # 读取图像
    image1 = image.read(image_path1)
    image2 = image.read(image_path2)
    if not image1 or not image2:
        print("Failed to read images")
        sys.exit(1)

    # 创建模型
    try:
        model_fd = nn.get_model_from_dir(nn.ModelType.SCRFD_DET_FACE, model_dir)
        model_fl = nn.get_model_from_dir(nn.ModelType.KEYPOINT_FACE_V2, model_dir)
        
        # 获取特征提取模型
        model_type = getattr(nn.ModelType, model_id_name)
        model_fe = nn.get_model_from_dir(model_type, model_dir)
    except Exception as e:
        print(f"Failed to create models: {e}")
        sys.exit(1)

    # 人脸检测
    input_images = [image1, image2]
    out_fd = []
    for img in input_images:
        result = model_fd.inference(img)
        out_fd.append(result)

    # 提取人脸裁剪和关键点
    face_crops_landmark = extract_crop_face_landmark(model_fl, input_images, out_fd)

    # 人脸对齐
    face_aligns = []
    for i, (face_crop, landmarks_meta) in enumerate(face_crops_landmark):
        face_align = face_crop_align(face_crop, landmarks_meta['landmarks'])
        face_aligns.append(face_align)
        image.write(face_align, f"face_align_{i}.jpg")

    # 可视化人脸裁剪和关键点
    visualize_face_crop(face_crops_landmark)

    # 检查是否有两张对齐的人脸
    if len(face_aligns) != 2:
        print("face_aligns size is not 2")
        sys.exit(1)

    # 特征提取
    features = []
    for i, face_align in enumerate(face_aligns):
        feature_result = model_fe.inference(face_align)
        
        if len(feature_result) > 0:
            # feature_result 本身就是特征向量
            feature_vec = np.array(feature_result)
            features.append(feature_vec)

    # 检查特征提取结果
    if len(features) != 2:
        print("face_aligns size is not 2")
        sys.exit(1)

    # 计算余弦相似度
    if len(features) == 2:
        # 转换为float类型避免溢出
        feature1 = features[0].astype(np.float32)
        feature2 = features[1].astype(np.float32)
        
        # 输出特征信息（对应C++版本）
        print(f"feature size: {len(feature1)}")
        print(f"feature_meta->embedding_type: {features[0].dtype}")
        
        # 计算相似度
        dot_product = np.dot(feature1, feature2)
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
            print(f"sim: {similarity}")
        else:
            print("Error: One of the feature vectors has zero norm")
    else:
        print("Failed to extract features from both images")