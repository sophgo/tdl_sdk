
from genericpath import exists
import cv2
import copy
import os
import sys
import glob

from nnfactory import NNFactory
if __name__ == "__main__":
    #usage: python3 test_face_alignment.py [model_dir] [img_file] 
    if len(sys.argv) != 3:
        print("usage: python3 test_face_alignment.py [model_dir] [img_file]")
        exit(0)
    modelpath = sys.argv[1]
    img_file = sys.argv[2]

    factory = NNFactory(modelpath)
    align_face = factory.align
    print("get detector & extractor")
    detector = factory.get_detector()

    
    img_file = img_file.encode('utf-8', errors='surrogateescape').decode('utf-8')
    img = cv2.imread(img_file)
    bboxes, landmarks, probs, _ = detector.predict(img)
    if len(bboxes) ==0:
        print("no face detected")
        exit(0)
    
    dst_d = './aligned_faces'
    os.makedirs(dst_d, exist_ok=True)
    for i in range(len(bboxes)):
        print("face:",i,"box:",bboxes[i],"landmarks:",landmarks[i],"score:",probs[i])
        box = bboxes[i]
        if box[2] - box[0] < 10:
            continue
        algin_img = align_face(img, bboxes[i], landmarks[i])

        dst_f = dst_d + '/face_{}.jpg'.format(i)
        cv2.imwrite(dst_f, algin_img)
        print("save aligned face to:",dst_f)
