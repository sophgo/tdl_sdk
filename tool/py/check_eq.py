import cv2
import sys
import numpy as np
from data_vis import decode_rgb_planar

def pytorch_preprocess(img,mean,scale):
    img1 = (img-mean)/scale
    return img1


if __name__ == "__main__":
    # const_img = np.ones((432,768,3))*100
    # const_img = const_img.astype(np.uint8)
    # cv2.imwrite(sys.argv[1],const_img)
    # sys.exit()
    vpss_rgb = decode_rgb_planar(sys.argv[1],is_bgr=True).astype(np.int8)
    print(vpss_rgb[0,0,:])
    img = cv2.imread(sys.argv[2])
    print(img[0,0,:])
    mean = np.array([127.5,127.5,127.5])
    scale = 128

    vpss_quant = 128.251
    pytorch_rgb = pytorch_preprocess(img,mean,scale)
    py_rgb = (pytorch_rgb*vpss_quant).astype(np.int8)
    print('pyrgb:',py_rgb[0,0,:])
    diff = np.abs(vpss_rgb - py_rgb)

    print('maxdiff:',np.max(diff),',mean:',np.mean(diff))