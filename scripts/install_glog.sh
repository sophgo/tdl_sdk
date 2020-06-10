#!/bin/bash
## Copyright 2018 Bitmain Inc.
## License
## Author Yangwen Huang <yangwen.huang@bitmain.com>

glog_libs=$(ldconfig -p | grep libglog)

if [ -z "$glog_libs" ]; then
    mkdir -p ~/glog cd ~/glog && \
        wget https://github.com/google/glog/archive/v0.3.5.zip && \
        unzip v0.3.5.zip && \
        rm v0.3.5.zip && \
        cd glog-0.3.5 && \
        mkdir build && \
        ./configure && \
        make -j8 && \
        sudo make install && \
        sudo ldconfig
else
    echo -e "Glog is installed."
fi;