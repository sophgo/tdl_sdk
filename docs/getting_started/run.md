# 运行指南

## 下载模型仓库

```shell
cd sdk_package
git clone https://github.com/sophgo/tdl_models.git
```

## BM1688运行示例

* 如果没有安装nfs客户端，则需要安装

    ```shell
    sudo apt install nfs-common
    ```

* 初次登录，挂载nfs
    如果没有安装nfs，则需要安装

    ```shell
    sudo apt-get install nfs-kernel-server
    ```

    ```shell
    ssh linaro@BOX_IP #BOX_IP为盒子IP
    cd /data
    mkdir sdk_package
    sudo mount -t nfs HOST_IP:/path/to/sdk_package /data/sdk_package #HOST_IP为编译主机IP
    ```

* 配置环境

    ```shell
    cd /data/sdk_package/tdl_sdk
    source scripts/envsetup.sh BM1688
    ```

* c++运行示例

    ```shell
    cd tdl_sdk/install/BM1688/bin

    ./sample_img_fd /data/sdk_package/tdl_models/bm1688/ /path/to/xx.jpg
    ```

* python运行示例

    ```shell
    cd tdl_sdk/install/BM1688/python

    python3 sample_fd.py /data/sdk_package/tdl_models/bm1688/scrfd_500m_bnkps_432_768.bmodel /path/to/xx.jpg
    ```

## BM1684X运行示例

* 初次登录，挂载nfs, 与BM1688相同

* 配置环境

    ```shell
    cd /data/sdk_package/tdl_sdk
    source scripts/envsetup.sh BM1684X
    ```

* c++运行示例

    ```shell
    cd tdl_sdk/install/BM1684X/bin

    ./sample_img_fd /data/sdk_package/tdl_models/bm1684x/ /path/to/xx.jpg
    ```

* python运行示例

    ```shell
    cd tdl_sdk/install/BM1684X/python

    python3 sample_fd.py /data/sdk_package/tdl_models/bm1684x/scrfd_500m_bnkps_432_768.bmodel /path/to/xx.jpg
    ```

## BM1684运行示例

* 初次登录，挂载nfs, 与BM1688相同

* 配置环境

    ```shell
    cd /data/sdk_package/tdl_sdk
    source scripts/envsetup.sh BM1684
    ```

* c++运行示例

    ```shell
    cd tdl_sdk/install/BM1684/bin

    ./sample_img_fd /data/sdk_package/tdl_models/bm1684/ /path/to/xx.jpg
    ```

* python运行示例

    ```shell
    cd tdl_sdk/install/BM1684/python

    python3 sample_fd.py /data/sdk_package/tdl_models/bm1684/scrfd_500m_bnkps_432_768.bmodel /path/to/xx.jpg
    ```

## CMODEL_CV181X运行示例

* 配置环境

    ```shell
    cd /data/sdk_package/tdl_sdk
    source scripts/envsetup.sh CMODEL_CV181X
    ```

* c++运行示例

    ```shell
    cd tdl_sdk/install/CMODEL_CV181X/bin

    ./sample_img_fd /data/sdk_package/tdl_models/cv181x/ /path/to/xx.jpg
    ```

## MARS3使用ive算子

* 配置环境
  * 克隆ive库，并切换至Mars3_dev分支(***改成自己的名字)

    ```shell
    git clone ssh://***@gerrit-ai.sophgo.vip:29418/cvitek/ive.git
    git checkout Mars3_dev
    ```

  * 拷贝算子库`ive/3rdparty/tpu_kernel_module/libtpu_kernel_module.so`到mars3板子`/mnt/data/tpu_files`下

  * 声明环境变量

    ```shell
    export BMRUNTIME_USING_FIRMWARE=/mnt/tpu_files/lib/libtpu_kernel_module.so
    ```

* 运行示例

  ```shell
  cd tdl_sdk/install/CV184X/bin
  ./sample_img_blend path/to/left_image path/to/right_image overlay_width
  ```
