

import cv2
import copy
import os
import sys
import glob
import numpy as np
from nnfactory import NNFactory
if __name__ == "__main__":
    #usage: python3 test_face_recog.py [model_dir] [face1_img_file] [face2_img_file] 
    if len(sys.argv) != 4:
        print("usage: python3 test_face_recog.py [model_dir] [face1_img_file] [face2_img_file]")
        exit(0)
    modelpath = sys.argv[1]
    face1_img_file = sys.argv[2]
    face2_img_file = sys.argv[3]

    factory = NNFactory(modelpath)
    align_face = factory.align
    print("get detector & extractor")
    detector = factory.get_detector("scrfd")


    img1 = cv2.imread(face1_img_file)
    img2 = cv2.imread(face2_img_file)
    bboxes1, landmarks1, probs1, _ = detector.predict(img1)
    bboxes2, landmarks2, probs2, _ = detector.predict(img2)
    if len(bboxes1) !=1  or len(bboxes2) !=1:
        print("face number not equal to 1,face1_img_file:{},face2_img_file:{}".format(len(bboxes1),len(bboxes2)))
        exit(0)
    
    aligned_face1 = align_face(img1, bboxes1[0], landmarks1[0])
    aligned_face2 = align_face(img2, bboxes2[0], landmarks2[0])

    print("aligned_face1:",aligned_face1.shape)
    print("aligned_face2:",aligned_face2.shape)
    
    extractor = factory.get_feature_extractor("BMFACER34")
    feat1 = extractor.predict_cropped([aligned_face1])[0]
    feat2 = extractor.predict_cropped([aligned_face2])[0]
    
    feat_sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    print("feature similarity:",feat_sim)
