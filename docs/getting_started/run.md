# 运行指南

## 下载模型仓库

```shell
cd sdk_package
git clone https://github.com/sophgo/tdl_models.git
```

## BM1688运行示例

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

## CMODEL_CVITEK运行示例

* 配置环境

    ```shell
    cd /data/sdk_package/tdl_sdk
    source scripts/envsetup.sh CMODEL_CVITEK
    ```

* c++运行示例

    ```shell
    cd tdl_sdk/install/CMODEL_CVITEK/bin

    ./sample_img_fd /data/sdk_package/tdl_models/cv181x/ /path/to/xx.jpg
    ```
