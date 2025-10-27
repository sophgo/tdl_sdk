import sys
import os
from tdl import nn, image
import cv2
import numpy as np

def visualize_objects(img_path, bboxes, save_path="object_detection.jpg"):
    """可视化目标检测结果"""
    img = cv2.imread(img_path)
    print(f"检测到 {len(bboxes)} 个目标")
    
    for i, obj in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, [obj['x1'], obj['y1'], obj['x2'], obj['y2']])
        class_id = obj['class_id']
        score = obj['score']
        class_name = obj.get('class_name', f'class_{class_id}')
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label = f"{class_id}:{score:.2f}"
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.putText(img, label, (center_x, center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite(save_path, img)
    print(f"保存图像到: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python3 sample_img_object_detection.py <model_id_name> <model_dir> <image_path> [threshold]")
        # 尚未加入对于不是检测模型的处理
        sys.exit(1)
    
    model_id_name = sys.argv[1]
    model_dir = sys.argv[2]
    image_path = sys.argv[3]
    threshold = float(sys.argv[4]) if len(sys.argv) == 5 else 0.5

    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        sys.exit(1)
    model_type = getattr(nn.ModelType, model_id_name)
    model = nn.get_model_from_dir(model_type, model_dir, device_id=0)

    # 读取图像
    img = image.read(image_path)
    
    # 执行推理
    outdatas = model.inference(img)

    expected_keys = {"class_id", "class_name", "score", "x1", "y1", "x2", "y2"}
    is_detection = (
        isinstance(outdatas, list) and
        isinstance(outdatas[0], dict) and
        set(outdatas[0].keys()) == expected_keys
    )

    if not is_detection:
        print("当前模型不是目标检测模型，输出内容：")
        print(outdatas)
        sys.exit(1)

    print(f"out_datas.size: {len(outdatas)}")

    for i, obj in enumerate(outdatas):
        print(f"obj_meta_index: {i}  "
                f"class: {obj['class_id']}  "
                f"score: {obj['score']:.2f}  "
                f"bbox: {obj['x1']:.2f} {obj['y1']:.2f} {obj['x2']:.2f} {obj['y2']:.2f}")

    visualize_objects(image_path, outdatas)

# input: python3 sample_img_object_detection.py <model_id_name> <model_dir> <image_path> [threshold]
# output: obj_meta_index: <index> class: <class_id> score: <score_value> bbox: <x1> <y1> <x2> <y2>
