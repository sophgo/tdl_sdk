#!/bin/bash
## Copyright 2018 Bitmain Inc.
## License
## Author Yangwen Huang <yangwen.huang@bitmain.com>

opencv_libs=$(ldconfig -p | grep libopencv)

if [ -z "$opencv_libs" ]; then
    mkdir -p ~/opencv cd ~/opencv && \
        wget https://github.com/Itseez/opencv/archive/3.4.0.zip && \
        unzip 3.4.0.zip && \
        rm 3.4.0.zip && \
        mv opencv-3.4.0 OpenCV && \
        cd OpenCV && \
        mkdir build && \
        cd build && \
        cmake ../ && \
        make -j8 && \
        sudo make install && \
        sudo ldconfig
else
    echo -e "OpenCV is installed."
fi;