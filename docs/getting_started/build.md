# 编译指南

## 基础准备

* 在ubuntu上安装准备编译软件

    ```shell
    sudo apt-get install gcc device-tree-compiler libssl-dev ssh bison flex
    ```

* 安装cmake，版本至少为3.16.3

    ```shell
    sudo apt-get install cmake
    ```

* 下载tdl_sdk

    ```shell
    mkdir sdk_package
    cd sdk_package
    git clone https://github.com/sophgo/tdl_sdk.git
    ```

### 181平台编译

* 下载sophpi

    ```shell
    cd sdk_package
    git clone https://github.com/sophgo/sophpi.git
    git checkout sg200x-evb
    ```

* 下载依赖仓库

    ```shell
    ./sophpi/scripts/repo_clone.sh --gitclone scripts/subtree.xml
    ```

* 执行编译

    ```shell
    cd tdl_sdk
    git checkout edge
    ./build.sh cv181x
    ```

### 186AH平台编译

* 下载固件等依赖仓库

    ```shell
        repo init -u https://github.com/sophgo/manifest.git -m release/all_repos.xml
        repo sync -j4
    ```

* 执行编译

    ```shell
    cd tdl_sdk
    git checkout edge
    ./build_tdl_sdk.sh cv186x

    ```

### BM1688平台编译

* 在[算能官网](https://developer.sophgo.com/site/index/material/92/all.html)下载sophonsdk_edge_v1.8_official_release压缩包
* 执行如下脚本抽取编译依赖

    ```sh
    cd tdl_sdk
    #会将依赖抽取到tdl_sdk/sophon_sdk文件夹下
    ./scripts/extract_sophon_sdk.sh
    ```

* 执行编译

    ```shell
    cd tdl_sdk
    ./build_tdl_sdk.sh bm168x
    ```

### BM1684X平台编译
