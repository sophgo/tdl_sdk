import glob
import sys


def scan_image_files(img_root,dst_list_file):
    if img_root[-1] != '/':
        img_root += '/'
    
    img_files = glob.glob(img_root+'/**/*.jpg')
    relative_files = [imgf[len(img_root):] for imgf in img_files]
    with open(dst_list_file,'w') as f:


        for rf in relative_files:
            f.write(rf+'\n')
    

if __name__ == "__main__":
    scan_image_files(sys.argv[1],sys.argv[2])