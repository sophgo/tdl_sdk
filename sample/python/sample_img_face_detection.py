import sys
import os
from tdl import nn, image
import cv2
import numpy as np

def visualize_faces(img_path, bboxes, save_path="face_detection.jpg"):
    img = cv2.imread(img_path)
    for i, face in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, [face['x1'], face['y1'], face['x2'], face['y2']])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if 'landmarks' in face:
            for (lx, ly) in face['landmarks']:
                cv2.circle(img, (int(lx), int(ly)), 3, (0, 255, 0), -1)
    cv2.imwrite(save_path, img)
    print(f"可视化结果已保存为: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_fd.py <model_path> <image_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    face_detector = nn.get_model(nn.ModelType.SCRFD_DET_FACE, model_path)
    img = image.read(img_path)
    bboxes = face_detector.inference(img)
    print('number of faces:', len(bboxes))
    for idx, face in enumerate(bboxes):
        print(f"face_{idx}: box=[{face['x1']:.1f}, {face['y1']:.1f}, {face['x2']:.1f}, {face['y2']:.1f}], "
              f"score={face['score']:.3f}")
        if 'landmarks' in face:
            for i, (lx, ly) in enumerate(face['landmarks']):
                print(f"    landmark_{i}: ({lx:.1f}, {ly:.1f})")
    visualize_faces(img_path, bboxes)