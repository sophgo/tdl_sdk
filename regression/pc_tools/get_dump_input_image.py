# coding: utf-8

from posixpath import join
import numpy as np
import cv2
import argparse
import os
import glob

def recover_origin_image(input_data, frame_format):
    if frame_format[0] == 12:     # PIXEL_FORMAT_YUV_PLANAR_420
        cvt_index = cv2.COLOR_YUV2BGR_I420
    elif frame_format[0] == 19:   # PIXEL_FORMAT_NV21
        cvt_index = cv2.COLOR_YUV2BGR_NV21
    elif frame_format[0] == 0:    # PIXEL_FORMAT_RGB_888
        cvt_index = cv2.COLOR_RGB2BGR
    else:
        raise ValueError('Unsupported frame format: {}'.format(frame_format))

    return cv2.cvtColor(input_data, cvt_index)

def recover_preprocessed_image(input_data, mean, factor):
    data = input_data.transpose(0, 2, 3, 1)
    images = []
    for batch in range(data.shape[0]):
        img = ((data[batch] + mean) / factor).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    return images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='dump folder')
    parser.add_argument('--output', required=True, help='dump folder')
    parser.add_argument('--origin-image', action='store_true', help='recover origin input frame')

    args = parser.parse_args()
    if not os.path.exists(args.input):
        raise ValueError('{} doesn\'t exists'.format(args.input))

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    elif not os.path.isdir(args.output):
        raise ValueError('{} is not a folder'.format(args.output))


    npz_files = sorted(glob.glob(os.path.join(args.input, '*.npz')))
    for dump_file in npz_files:
        npz = np.load(dump_file)
        for k in npz.keys():
            output_prefix = os.path.basename(dump_file).replace('.npz', '')

            # recover original image
            if k.startswith('input_orig_') and not k.endswith('_frame_format') and args.origin_image:
                im = recover_origin_image(npz[k], npz[k + '_frame_format'])
                output_path = os.path.join(args.output, output_prefix + '_' + k + '.jpg')
                print('save {}'.format(output_path))
                cv2.imwrite(output_path, im)
            
            # recover preprocessed image
            if k.startswith('input_data_') and not k.endswith('_mean') and not k.endswith('_factor') and not k.endswith('_qscale'):
                images = recover_preprocessed_image(npz[k], npz[k + '_mean'], npz[k + '_factor'])
                for batch_idx, im in enumerate(images):
                    output_path = os.path.join(args.output, output_prefix + '_' + k + '_batch_{}.jpg'.format(batch_idx))
                    print('save {}'.format(output_path))
                    cv2.imwrite(output_path, im)