import sys
from tdl import nn,image
import numpy as np
import cv2

model_id_mapping = {
    "FEATURE_BMFACE_R34": nn.ModelType.FEATURE_BMFACE_R34,
    "FEATURE_BMFACE_R50": nn.ModelType.FEATURE_BMFACE_R50,
    "FEATURE_CVIFACE": nn.ModelType.FEATURE_CVIFACE,
}

def extract_feature(img_path, extractor):
    # 读取图片
    img = image.read(img_path)
    # 提取特征
    feature = extractor.inference(img)
    return feature

def calculate_similarity(feat1, feat2):
    # 特征归一化
    feat1 = feat1 / np.linalg.norm(feat1)
    feat2 = feat2 / np.linalg.norm(feat2)
    # 计算相似度
    similarity = np.sum(feat1 * feat2)
    return similarity

def detect_and_align_face(img_path, detector):
    # 方式一
    # img_numpy = cv2.imread(img_path)
    # img = image.from_numpy(img_numpy)
    # 方式二
    img = image.read(img_path)
    bboxes = detector.inference(img)
    if len(bboxes) == 0:
        print("No face detected")
        return None
    face_info = bboxes[0]
    landmarks = face_info['landmarks']

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

    aligned_img = image.align_face(img, src_landmarks, dst_landmarks, 5)
    return aligned_img




if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python sample_face_feature.py <feature_extraction_model_id_name> <model_dir> \
              <image_path1> <image_path2>")
        print("feature_extraction_model_id_name: ", list(model_id_mapping.keys()))
        sys.exit(1)
    feature_extraction_model_id_name = sys.argv[1]
    model_dir = sys.argv[2]
    img_path1 = sys.argv[3]
    img_path2 = sys.argv[4]

    face_detector = nn.get_model_from_dir(nn.ModelType.SCRFD_DET_FACE, model_dir)

    model_type = model_id_mapping[feature_extraction_model_id_name]
    feature_extractor = nn.get_model_from_dir(model_type, model_dir)

    aligned_img1 = detect_and_align_face(img_path1, face_detector)
    aligned_img2 = detect_and_align_face(img_path2, face_detector)

    image.write(aligned_img1, "aligned_img1.jpg")
    image.write(aligned_img2, "aligned_img2.jpg")


    feat1 = feature_extractor.inference(aligned_img1)
    feat2 = feature_extractor.inference(aligned_img2)

    similarity = calculate_similarity(feat1, feat2)
    print(f"similarity: {similarity:.4f}")