# 编译指南

## 编译主机准备

* 编译主机为X86架构，操作系统为Ubuntu
* 在编译主机上安装编译软件

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

* 下载第三方依赖

    ```shell
    cd tdl_sdk
    ./scripts/download_thirdparty.sh
    ```

* nfs配置

  * 安装nfs服务端

    ```shell
    sudo apt-get install nfs-kernel-server
    ```

  * 配置nfs服务端

    ```shell
    sudo vim /etc/exports
    # 添加如下内容，修改/path/to/sdk_package为sdk_package真实路径
    /path/to/sdk_package *(rw,sync,no_subtree_check,no_root_squash)
    ```

  * 重启nfs服务

    ```shell
    sudo systemctl restart nfs-server
    ```

## 编译

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
    ./scripts/extract_sophon_sdk.sh /path/to/sophonsdk_edge_v1.8_offical_release.zip
    ```

* 执行编译

    ```shell
    cd tdl_sdk
    ./build_tdl_sdk.sh bm168x
    ```

#### Python导出包编译

* 到目标盒子上安装Python环境

    ```shell
    sudo apt-get install python3.8 python3-pip
    pip3 install pybind11
    ```

* 挂载nfs目录

    ```shell
    cd /data
    mkdir sdk_package
    #HOST_IP为编译主机IP
    sudo mount -t nfs HOST_IP:/path/to/sdk_package /data/sdk_package
    ```

* 在盒子上收集Python编译依赖

    ```shell
    cd /data/sdk_package/tdl_sdk/
    chmod +x scripts/pack_python_depends.sh
    ./scripts/pack_python_depends.sh
    ```

* 到X86主机上重新执行编译

    ```shell
    cd /data/sdk_package/tdl_sdk/
    ./build.sh bm186x
    ```

### BM1684X平台编译
