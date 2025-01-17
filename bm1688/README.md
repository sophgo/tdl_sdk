
## 环境准备

* SDK下载
  * [sophonsdk_edge_v1.8_official_release 下载](https://developer.sophgo.com/site/index/material/92/all.html)

* 固件刷机
  * V1.8.0->sophon-img->sdcard.tgz，解压到存储卡刷机

* 安装依赖
  * 安装libsophon0.4.10.deb、opencv_1.8.0_arm64.deb、ffmpeg_1.8.0_arm64.deb获得库文件，会被安装到/opt/sophon/目录下
  * 安装opencv-dev_1.8.0_arm64.deb及ffmpeg-dev_1.8.0_arm64.deb获得头文件，会被安装到/opt/sophon/目录下

## 编译

### c++库编译

```shell
cd /bm1688
mkdir build
cd build
cmake .. -DPLATFORM_BUILD_TYPE=native_arm -DUSE_CHIP_TYPE=BM1688 -DUSE_FFMPEG=ON
make -j4
```

### python接口编译

```shell
cd /bm1688/python
mkdir build
cd build
cmake .. -DPLATFORM_BUILD_TYPE=native_arm -DUSE_CHIP_TYPE=BM1688
make -j4
```

## demo使用

### 模型下载

```sh
git clone https://github.com/sophgo/tdl_models.git
```

使用说明：

* 使用tdl_models/bm1688下的模型
* face detection (cssd)：需要`cssd_int8.bmodel`模型、`cssd_prior_box.bin`文件
* face detection （scrfd）需要`scrfd_500m_bnkps_432_768_cv186x.bmodel`模型
* face landmark (det3): 需要`landmark_det3.bmodel`模型
* face feature 需要bmface系列模型

假如模型文件是通过网络挂载到se9盒子上，需要执行如下命令

```sh
#登录se9盒子
#ssh linaro@xxx.xxx.xxx.xxx
sudo chown -R linaro:linaro  /path/to/tdl_models/bm1688
```

### c++ demo使用

```sh
#登录se9盒子

cd tdl_sdk/bm1688/build
export LD_LIBRARY_PATH=/opt/sophon/sophon-ffmpeg_1.8.0/lib:/opt/sophon/sophon-opencv_1.8.0/lib:/opt/sophon/libsophon-0.4.10/lib/:$(pwd)/lib

#人脸检测
./bin/demo_detector /path/to/tdl_models/bm1688 0.6 ../data/Abel_Pacheco_0001.jpg


```

### python demo使用

```sh

cd tdl_sdk/bm1688/python/tests
export PYTHONPATH=$PYTHONPATH:$(pwd)/../

# 人脸检测
python3 test_face_detection.py /path/to/tdl_models/bm1688 ../../data/test_2.jpg

# 人脸对齐
python3 test_face_alignment.py /path/to/tdl_models/bm1688 ../../data/test_2.jpg

# 人脸特征提取比对
python3 test_face_recog.py /path/to/tdl_models/bm1688 ../../data/Abel_Pacheco_0001.jpg ../../data/Abel_Pacheco_0004.jpg

```
