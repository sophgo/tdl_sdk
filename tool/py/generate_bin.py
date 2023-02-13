import numpy as np

import cv2
import sys
import glob
def convert_image_to_bin(src_file,dst_file,plannar):
    img = cv2.imread(src_file)
    h,w,_ = img.shape
    wh = np.array([w,h]).astype(np.int32)
    if plannar:
        img = np.transpose(img,[2,0,1])
        assert(img.shape[0] == 3)

    print('w:',w,',h:',h)
    # for i in range(100):
    #     print(img[0,0,i])
    dstf = open(dst_file,'wb')
    dstf.write(wh.tobytes())
    dstf.write(img.tobytes())

    dstf.close()

if __name__ == "__main__":
    convert_image_to_bin(sys.argv[1],sys.argv[2],True)



