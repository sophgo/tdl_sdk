import cv2
import sys
def draw_image(img,boxes,kpts=None):

    for box in boxes:
        ib = [int(v) for v in box]
        cv2.rectangle(img,tuple(ib[:2]),tuple(ib[2:4]),(0,0,255),2)
    if kpts is None:
        return
    for kpt in kpts:
        ipt = [int(v) for v in kpt]
        for j in range(len(ipt)//2):
            cv2.circle(img,(ipt[2*j],ipt[2*j+1]),2,(0,0,255),-1)


if __name__ == "__main__":
    imgf = sys.argv[1]
    boxes = boxes=[[388.333,37.2222,435.694,108.056],[185.556,99.7222,235,166.667],[332.153,10.3472,364.583,51.6667],[219.097,0,247.778,22.1528]]
    kpts=[[403.611,65.3125,425.764,66.3542,415.729,80.625,404.132,90.1389,422.083,91.1111],[203.333,121.806,224.931,119.167,218.611,135.486,208.576,149.167,224.931,147.083],[342.257,22.6563,357.292,23.0556,350.486,31.0764,343.663,38.8889,356.319,39.0972],[226.892,0.243056,240.556,-0.347222,234.115,6.07639,228.507,12.1007,239.549,11.7014]]

    img = cv2.imread(imgf)
    print('imgshape:',img.shape,img.dtype)
    draw_image(img,boxes,kpts)
    cv2.imwrite("xx.jpg",img)
    cv2.imshow('box',img)
    cv2.waitKey(0)