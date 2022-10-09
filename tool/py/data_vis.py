import numpy as np

import cv2
import sys
import json

def decode_rgb_planar(binf, s=1):
    data = np.fromfile(binf, dtype=np.uint8)
    print('datalen:', data.shape)
    datawh = data[:8].view('int32')

    img = data[8:]
    print('imgshape:', img.shape)
    img = img.reshape(3, datawh[1], -1).astype(np.uint8)
    print(img.shape)
    print('datawh:', datawh)
    rgb_pack = np.transpose(img, [1, 2, 0])
    bgr_pack = rgb_pack[:, :, ::-1].copy()
    
    return bgr_pack

# pY[i + j * vFrame->u32Stride[0]] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;

# if (j % 2 == 0 && i % 2 == 0) {
# pU[width / 2 + height / 2 * vFrame->u32Stride[1]] = ((-38 * r - 74 * g + 112 * b) >> 8) + 128;
# pV[width / 2 + height / 2 * vFrame->u32Stride[2]] = ((112 * r - 94 * g - 18 * b) >> 8) + 128;
# }

def rgb2yuv(r,g,b):
    y = (66*r+129*g+25*b)/256 + 16
    u = (-38*r-74*g+112*b)/256 + 128
    v = (112*r-94*g-18*b)/256 + 128
    return y,u,v
def yuv2rgb(y,u,v):
    pass

def decode_nv12(binf,size=None):

    data = np.fromfile(binf, dtype=np.uint8)
    print('datalen:', data.shape)
    if size is None:
        datawh = data[:8].view('int32')
        width = datawh[0]
        height = datawh[1]
        input = data[8:]
    else:
        width = size[0]
        height = size[1]
        input = data

    print('wh:',[width,height],',shape:',input.shape)

    nv_off = int(width * height)


    output = np.zeros((width*height*3,), dtype=np.uint8)
    print('outshape:',output.shape,',input:',input.shape)
    yv, xv = np.meshgrid(np.arange(height), np.arange(width))
    print('yvhsape:',yv.shape)
    yv = yv.flatten()
    xv = xv.flatten()
    nv_index = (yv//2)*width + xv - xv%2
    uidx = nv_index.astype(np.int32) + nv_off
    yidx = np.arange(nv_off)
    vidx = uidx+1
    y = input[yidx]
    u = input[uidx]
    v = input[vidx]
    r = y + ((351 * (v - 128)) >> 8)  # r
    g = y - ((179 * (v - 128) + 86 * (u - 128)) >> 8)  # g
    b = y + ((443 * (u - 128)) >> 8)  # b

    # c = (y-16) * 298
    # d = u - 128
    # e = v - 128

    # r = (c + 409 * e + 128) // 256
    # g = (c - 100 * d - 208 * e + 128) // 256
    # b = (c + 516 * d + 128) // 256

    img = np.vstack((r,g,b)).reshape(3,height,width).transpose([1,2,0])
    img = np.clip(img,0,255).astype(np.uint8)[:,:,::-1].copy()

    return img
def load_algo_result(txtf):
    with open(txtf,'r') as f:
        lines = f.readlines()
    #only parse box and score
    datas = []
    for l in lines:
        if len(l)<10:
            print('not ok data:',l)
            continue
            
        l = '{'+l.strip('\n')+'}'
        print('l:',l)
        datai = json.loads(l)
        datas.append(datai)

    return datas


def visualize_result(imgf,resultf):
    img = decode_nv12(imgf)
    datas = load_algo_result(resultf)

    for di in datas:
        box = np.array(di['x1y1x2y2']).astype(np.int32).tolist()
        cv2.rectangle(img,tuple(box[:2]),tuple(box[2:4]),(0,255,0),2)
    
    cv2.imshow('img',img)
    cv2.waitKey(0)


if __name__ == "__main__":
    # visualize_result(sys.argv[1],sys.argv[2])
    img = decode_nv12(sys.argv[1],[384,256])
    cv2.imshow("img",img)
    cv2.waitKey(0)

        
        