import sys
import os
from tdl import nn,image

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_face_landmark.py <model_path> <image_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    face_landmark = nn.FaceLandmark(nn.ModelType.KEYPOINT_FACE_V2, model_path)
    img = image.read(img_path)
    face_landmarks = face_landmark.inference(img)
    print(face_landmarks)