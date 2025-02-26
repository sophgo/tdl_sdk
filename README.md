# TDL_SDK 公版深度学习软件包（Turnkey Deep Learning SDK）

该文档介绍如何编译及使用tdl_sdk。

该版本已完成对

- v4.1.0
- v4.2.0
- athena2
- mars3

四个版本的SDK支持，只需将tdl_sdk放置在对应版本的项目根路径中便可使用。

## 下载

- 在空文件夹中下载cvi_manifest来辅助下载不同版本的SDK

``` shell
mkdir sdk_package
cd sdk_package
# 内部从gerrit上clone
git clone ssh://user.name@gerrit-ai.sophgo.vip:29418/cvi_manifest
```

- 根据不同的芯片，先下载对应的项目包

``` shell
# v4.1.0
./cvi_manifest/cvitek_repo_clone.sh --gitclone cvi_manifest/golden/cv181x_cv180x_v4.1.0.xml

# v4.2.0
./cvi_manifest/cvitek_repo_clone.sh --gitclone cvi_manifest/golden/cv181x_cv180x_v4.2.0.xml

# athena2 edge
./cvi_manifest/cvitek_repo_clone.sh --gitclone cvi_manifest/athena2/master/edge.xml

# athena2 device
./cvi_manifest/cvitek_repo_clone.sh --gitclone cvi_manifest/athena2/master/device.xml

# mars3
./cvi_manifest/cvitek_repo_clone.sh --gitclone cvi_manifest/mars3.xml
```

下载完整包项目SDK之后，若根路径下没有tdl_sdk，可执行以下命令下载tdl_sdk：

``` shell
# 项目根路径下载tdl_sdk
./cvi_manifest/cvitek_repo_clone.sh --gitclone cvi_manifest/tdl_sdk.xml
```

- 下载完成后代码都在sdk_package目录下

## 编译

### 181x/186x编译

- 编译依赖库

``` shell
source build/envsetup_soc.sh

# 181x
defconfig 181x
defconfig cv18xxx_xxx_xxx #cv1811h_wevb_0007a_spinor

# 186x
defconfig 186x
defconfig cv18xxx_xxx_xxx #cv186x_cv186x_v1.0.0

export TPU_REL=1

clean_all && build_all
```

- 编译tdl_sdk

``` shell
cd tdl_sdk

# 181x
# 编译 modules
./build_tdl_sdk.sh

# 编译 samples
./build_tdl_sdk.sh sample

# 编译 modules + samples
./build_tdl_sdk.sh all

# 186x

build_ai_sdk
```

## bm1688编译

``` shell
cd tdl_sdk
git checkout bm1688
ln -s /data/fuquan.ke/nfsuser/sophon_sdk/ sophon_sdk

# 编译
scripts/build_bm1688.sh
```

## 执行

- 181x

    ``` shell

    ssh root@172.25.3.45
    cd /mnt/data
    mkdir sdk_package
    # 挂载build服务器
    mount -t nfs 10.80.39.3:/path/to/build/sdk_package ./sdk_package -o nolock

    cd /mnt/data/sdk_package/install/soc_cv1811h_wevb_0007a_spinor
    export LD_LIBRARY_PATH=/mnt/data/sdk_package/install/soc_cv1811h_wevb_0007a_spinor/tpu_musl_riscv64/cvitek_tpu_sdk/lib:/mnt/data/sdk_package/cvi_mpi/lib:/mnt/data/sdk_package/cvi_mpi/lib/3rd:/mnt/data/sdk_package/tdl_sdk/installCV181X/lib:/mnt/data/sdk_package/tdl_sdk/installCV181X/sample/3rd/opencv/lib

    #开启log
    /sbin/syslogd -l 8 -s 2048 -O /mnt/data/log.txt

    cd /mnt/data/sdk_package/tdl_sdk/installCV181X/regression
    ./daily_regression.sh -m /path/to/ai_models_cv181x/ -d /path/to/aisdk_daily_regression/ -a /path/to/tdl_sdk/regression/assets_181x/
    ```

- CV186AH

    ``` shell
    ssh root@172.25.3.92

    cd /mnt/data
    mkdir sdk_package
    # 挂载build服务器
    mount -t nfs 10.80.39.3:/path/to/build/sdk_package ./sdk_package -o nolock

    # 进入regression目录
    cd /mnt/data/sdk_package/install/soc_cv186ah_wevb_emmc


    export LD_LIBRARY_PATH=$(pwd)/tpu_64bit/cvitek_tpu_sdk/libsophon-0.4.9/lib:$(pwd)/tpu_64bit/cvitek_ai_sdk/lib:$(pwd)/tpu_64bit/cvitek_ai_sdk/sample/3rd/rtsp/lib/:$(pwd)/rootfs/mnt/system/usr/lib/:$(pwd)/rootfs/mnt/system/usr/lib/3rd:$(pwd)/tpu_64bit/cvitek_ai_sdk/sample/3rd/opencv/lib/

    cd tpu_64bit/cvitek_ai_sdk/regression

    rm /mnt/data/log.txt
    /sbin/syslogd -l 8 -s 2048 -O /mnt/data/log.txt

    ./daily_regression.sh -m /path/to/ai_models_cv186x/ -d /path/to/aisdk_daily_regression/ -a /path/to/tdl_sdk/regression/assets_186x/
    ```

- bm1688盒子

    ``` shell

    ssh linaro@172.25.3.56
    cd tdl_sdk
    cd build1688/install/regression

    export LD_LIBRARY_PATH=/opt/sophon/sophon-ffmpeg_1.8.0/lib:/opt/sophon/sophon-opencv_1.8.0/lib:/opt/sophon/libsophon-0.4.10/lib/:$(pwd)/../lib
    ./daily_regression.sh -m /path/to/ai_models_cv186x/ -d /path/to/aisdk_daily_regression/ -a /path/to/tdl_sdk/regression/assets_186x/
    ```
