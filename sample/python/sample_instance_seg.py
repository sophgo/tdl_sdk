import sys
import os
import cv2
import numpy as np
from tdl import nn, image

def visualize_mask(segmentation_result, output_name="yolov8_seg_mask.png"):
    if len(segmentation_result) == 0:
        print("没有检测到对象")
        return False
    
    combined_mask = None
    
    for i, result in enumerate(segmentation_result):
        if isinstance(result, dict):
            mask_height = result.get('mask_height')
            mask_width = result.get('mask_width')
            
            # 查找嵌套的掩码数据
            if 'bboxes_seg' in result and isinstance(result['bboxes_seg'], list):
                for j, bbox_data in enumerate(result['bboxes_seg']):
                    if isinstance(bbox_data, dict) and 'mask' in bbox_data:
                        mask_data = bbox_data['mask']
                        
                        if mask_height and mask_width and len(mask_data) == mask_height * mask_width:
                            obj_mask = np.array(mask_data, dtype=np.float32)
                            obj_mask = obj_mask.reshape((mask_height, mask_width))
                            obj_mask = (obj_mask * 255).astype(np.uint8)
                            
                            if combined_mask is None:
                                combined_mask = obj_mask.copy()
                            else:
                                if obj_mask.shape == combined_mask.shape:
                                    combined_mask = cv2.bitwise_or(combined_mask, obj_mask)
                        
    
    if combined_mask is not None:
        cv2.imwrite(output_name, combined_mask)
        print(f"掩码图像已保存到: {output_name}")
        return True
    else:
        print("未找到可用的掩码数据")
        return False

def visualize_mask_outline(segmentation_result, img_path, output_name="yolov8_seg_outline.jpg"):
    if len(segmentation_result) == 0:
        print("没有检测到对象")
        return False
    
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"无法读取图像: {img_path}")
        return False
    
    image_height, image_width = original_img.shape[:2]
    
    first_result = segmentation_result[0]
    mask_height = first_result.get('mask_height')
    mask_width = first_result.get('mask_width')
    
    if mask_height is None or mask_width is None:
        print("未找到掩码尺寸信息")
        return False
    
    result_img = original_img.copy()
    mask_points_found = False
    
    for i, result in enumerate(segmentation_result):
        if isinstance(result, dict):
            # 查找嵌套的掩码数据
            if 'bboxes_seg' in result and isinstance(result['bboxes_seg'], list):
                for j, bbox_data in enumerate(result['bboxes_seg']):
                    if isinstance(bbox_data, dict) and 'mask' in bbox_data:
                        mask_data = bbox_data['mask']
                        
                        if len(mask_data) == mask_height * mask_width:
                            # 创建当前对象的掩码图像
                            obj_mask = np.array(mask_data, dtype=np.float32)
                            obj_mask = obj_mask.reshape((mask_height, mask_width))
                            obj_mask = (obj_mask * 255).astype(np.uint8)
                            
                            contours, _ = cv2.findContours(obj_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if len(contours) > 0:
                                longest_contour = max(contours, key=cv2.contourArea)
                                
                                if len(longest_contour) >= 1:
                                    ratio_height = mask_height / image_height
                                    ratio_width = mask_width / image_width
                                    
                                    if ratio_height > ratio_width:
                                        source_x_offset = 0
                                        source_y_offset = int((mask_height - image_height * ratio_width) / 2)
                                    else:
                                        source_x_offset = int((mask_width - image_width * ratio_height) / 2)
                                        source_y_offset = 0
                                    
                                    source_region_height = mask_height - 2 * source_y_offset
                                    source_region_width = mask_width - 2 * source_x_offset
                                    
                                    height_scale = image_height / source_region_height
                                    width_scale = image_width / source_region_width
                                    
                                    transformed_points = []
                                    for point in longest_contour:
                                        x, y = point[0]
                                        transformed_x = (x - source_x_offset) * width_scale
                                        transformed_y = (y - source_y_offset) * height_scale
                                        transformed_points.append([int(transformed_x), int(transformed_y)])
                                    
                                    if len(transformed_points) > 1:
                                        transformed_points = np.array(transformed_points, dtype=np.int32)
                                        cv2.polylines(result_img, [transformed_points], True, (0, 255, 0), 2, cv2.LINE_AA)
                                        mask_points_found = True
    
    cv2.imwrite(output_name, result_img)
    if mask_points_found:
        print(f"轮廓图像已保存到: {output_name}")
        return True
    else:
        print("未找到可用的掩码点数据")
        return False

def visualize_object_detection(img_path, segmentation_result, output_name="yolov8_seg_box.jpg"):
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"无法读取图像: {img_path}")
        return False
    
    box_found = False
    for i, result in enumerate(segmentation_result):
        if isinstance(result, dict):
            # 查找嵌套的边界框数据
            if 'bboxes_seg' in result and isinstance(result['bboxes_seg'], list) and len(result['bboxes_seg']) > 0:
                for j, bbox_data in enumerate(result['bboxes_seg']):
                    if isinstance(bbox_data, dict) and 'x1' in bbox_data:
                        x1 = int(bbox_data['x1'])
                        y1 = int(bbox_data['y1'])
                        x2 = int(bbox_data['x2'])
                        y2 = int(bbox_data['y2'])
                        
                        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                        class_id = bbox_data.get('class_id', 0)
                        score = bbox_data.get('score', 0)
                        class_name = bbox_data.get('class_name', '')
                        label = f"{class_name}:{score:.2f}"
                        cv2.putText(original_img, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        box_found = True
                        print(f"边界框: ({x1}, {y1}, {x2}, {y2})")
            else:
                # 原有的查找逻辑作为备选
                possible_coord_keys = {
                    'x1': ['x1', 'left', 'x_min', 'xmin'],
                    'y1': ['y1', 'top', 'y_min', 'ymin'], 
                    'x2': ['x2', 'right', 'x_max', 'xmax'],
                    'y2': ['y2', 'bottom', 'y_max', 'ymax']
                }
                
                coords = {}
                for coord_name, possible_names in possible_coord_keys.items():
                    for name in possible_names:
                        if name in result:
                            coords[coord_name] = result[name]
                            break
                
                if len(coords) == 4:
                    x1 = int(coords['x1'])
                    y1 = int(coords['y1']) 
                    x2 = int(coords['x2'])
                    y2 = int(coords['y2'])
                    
                    cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    class_id = result.get('class_id', result.get('cls', result.get('class', 0)))
                    score = result.get('score', result.get('confidence', result.get('conf', 0)))
                    label = f"{class_id},{score:.2f}"
                    cv2.putText(original_img, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    box_found = True
                    print(f"找到边界框: ({x1}, {y1}, {x2}, {y2})")
    
    cv2.imwrite(output_name, original_img)
    if box_found:
        print(f"检测框图像已保存到: {output_name}")
        return True
    else:
        print("未找到边界框信息")
        return False

def print_segmentation_details(segmentation_result, image_height=None, image_width=None):
    print("\nSegmentation details:")
    print("=" * 50)
    
    print(f"检测到 {len(segmentation_result)} 个对象")
    
    for i, result in enumerate(segmentation_result):
        print(f"\n对象 {i}:")
        if isinstance(result, dict):
            # 打印所有字段信息
            for key, value in result.items():
                if isinstance(value, (list, np.ndarray)) and len(value) > 10:
                    print(f"  {key}: {type(value)} (长度: {len(value)})")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  值: {result} (类型: {type(result)})")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_instance_seg.py <model_dir> <image_path>")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    img_path = sys.argv[2]
    
    instance_segmentator = nn.get_model_from_dir(nn.ModelType.YOLOV8_SEG_COCO80, model_dir)
    
    img = image.read(img_path)
    
    instance_segmentation = instance_segmentator.inference(img)
    
    print("实例分割原始结果:")
    print(instance_segmentation)
    
    original_img = cv2.imread(img_path)
    if original_img is not None:
        image_height, image_width = original_img.shape[:2]
        print_segmentation_details(instance_segmentation, image_height, image_width)
    else:
        print_segmentation_details(instance_segmentation)
    
    if len(instance_segmentation) > 0:
        visualize_mask(instance_segmentation, "yolov8_seg_mask.png")
        visualize_mask_outline(instance_segmentation, img_path, "yolov8_seg_outline.jpg")
        visualize_object_detection(img_path, instance_segmentation, "yolov8_seg_box.jpg")
    else:
        print("未检测到任何对象")
        empty_mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.imwrite("yolov8_seg_mask.png", empty_mask)
        print("已创建空的掩码图像: yolov8_seg_mask.png")
        
        original_img = cv2.imread(img_path)
        if original_img is not None:
            cv2.imwrite("yolov8_seg_outline.jpg", original_img)
            cv2.imwrite("yolov8_seg_box.jpg", original_img)
            print("已创建默认输出图像: yolov8_seg_outline.jpg, yolov8_seg_box.jpg")
