import sys
import os
from tdl import nn,image

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_fd.py <model_path> <image_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    face_detector = nn.FaceDetector(nn.ModelType.FD_SCRFD, model_path)
    img = image.read(img_path)
    bboxes = face_detector.inference(img)
    print('number of faces: ', len(bboxes))
    print(bboxes)

    # print(tdl_root)
    