import cv2
import numpy as np
import os
import time


import sys



if __name__ == "__main__":
    #usage: python3 test_fd.py [model_dir] [img_file] 
    if len(sys.argv) != 3:
        print("usage: python3 test_fd.py [model_dir] [img_file]")
        exit(0)
    modelpath = sys.argv[1]
    img_file = sys.argv[2]

    from nnfactory import NNFactory

   
    factory = NNFactory(modelpath)
    img = cv2.imread(img_file)

    detector = factory.get_detector("scrfd")
    extractor =factory.get_feature_extractor("BMFACER34")

    bboxes, landmarks, probs, _ = detector.predict(img)
    # print("face detection result:",bboxes, landmarks, probs)

    for i in range(len(bboxes)):
        print("face:",i,"box:",bboxes[i],"landmarks:",landmarks[i],"score:",probs[i])
        box = bboxes[i] #first face box for batch 0 image
        start_point = (int(box[0]), int(box[1]))  # 矩形的左上角
        end_point = (int(box[2]), int(box[3]))  # 矩形的右下角
        color = (0, 0, 255)  
        cv2.rectangle(img, start_point, end_point, color, 3)
        landmark = landmarks[i]
        num_landmark = len(landmark)//2
        for j in range(num_landmark):
            cv2.circle(img, (int(landmark[j]), int(landmark[j+num_landmark])), 2, (0, 0, 255), 3)
    cv2.imwrite('face_detection_result.jpg', img)

   


    
