# 编译指南

## 基础准备

1. 在ubuntu上安装准备编译软件：sudo apt-get install gcc device-tree-compiler libssl-dev ssh bison flex
2. 安装cmake，版本至少为3.16.3

### 181平台编译
1. 新建并进入sdk_package文件夹，克隆[sophpi](https://github.com/sophgo/sophpi)仓库，切换至sg200x-evb分支
2. 执行以下指令 下载源码：
```shell
    ./sophpi/scripts/repo_clone.sh --gitclone scripts/subtree.xml
```
3. 将tdl_sdk 仓库切换至edge分支
4. 执行以下步骤进行编译：
```shell
    cd tdl_sdk
    ./build.sh cv181x
```

### 186AH平台编译

1. 新建并进入sdk_package文件夹，执行以下指令下载源码：
```shell
    repo init -u https://github.com/sophgo/manifest.git -m release/all_repos.xml
    repo sync -j4
```
2. 将tdl_sdk 仓库切换至edge分支
3. 执行以下步骤进行编译：
```shell
    cd tdl_sdk
    ./build.sh cv186x
```

### BM1688平台编译

1. 克隆[tdl_sdk](https://github.com/sophgo/tdl_sdk)仓库，切换至edge分支
2. 在[算能官网](https://developer.sophgo.com/site/index/material/92/all.html)下载sophonsdk_edge_v1.8_official_release压缩包解压，得到V1.8_sophonsdk_edge_v1.8_ofical_release
3. 在tdl_sdk上新建sophon_sdk文件夹，解压V1.8_sophonsdk_edge_v1.8_ofical_release/sophon-img/libsophon_soc_0.4.10_aarch64.tar.gz，将其中的libsophon-0.4.10目录拷贝到sophon_sdk文件夹下。
4. 解压V1.8_sophonsdk_edge_v1.8_ofical_release/sophon_media/sophon-media-soc_1.8.0_aarch64.tar.gz，将其中的sophon-opencv_1.8.0、sophon-ffmpeg_1.8.0拷贝到sophon_sdk文件夹下
5. 解压V1.8_sophonsdk_edge_v1.8_ofical_release/sophon-img/bsp-debs/sophon-soc-libisp-dev_1.0.0_arm64.deb得到sophon-soc-libisp-dev_1.0.0，解压V1.8_sophonsdk_edge_v1.8_ofical_release/sophon-img/sophon-soc-libisp_1.0.0_arm64.deb得到sophon-soc-libisp_1.0.0，将sophon-soc-libisp-dev_1.0.0中的内容覆盖到sophon-soc-libisp_1.0.0，将sophon-soc-libisp_1.0.0拷贝到sophon_sdk文件夹下
6. 克隆[host-tools](https://github.com/sophgo/host-tools) 仓库，放在tdl_sdk同级目录
7. 执行以下步骤进行编译：
```shell
    cd tdl_sdk
    ./build.sh bm168x
```


### BM1684X平台编译

