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

* 下载tdl_sdk和host-tools

    ```shell
    mkdir sdk_package
    cd sdk_package
    git clone https://github.com/sophgo/host-tools.git
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

支持的目标平台有：

* CV181X
* CV186X
* BM1688
* BM1684X
* CMODEL

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
    ./build_tdl_sdk.sh CV181X
    #再次编译
    ./build_tdl_sdk.sh all
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
    ./build_tdl_sdk.sh CV186X
    #再次编译
    ./build_tdl_sdk.sh all
    ```

### BM1688平台编译

* 在[算能官网](https://developer.sophgo.com/site/index/material/92/all.html)下载sophonsdk_edge_v1.8_official_release压缩包
* 执行如下脚本抽取编译依赖

    ```sh
    cd tdl_sdk
    #会将依赖抽取到tdl_sdk/dependency/BM1688文件夹下
    ./scripts/extract_sophon_sdk.sh /path/to/sophonsdk_edge_v1.8_offical_release.zip BM1688
    ```

* 执行编译

    ```shell
    cd tdl_sdk
    ./build_tdl_sdk.sh BM1688
    ```

### BM1684X平台编译

* 在[算能官网](https://developer.sophgo.com/site/index/material/88/all.html)下载SDK-24.04.01的压缩包

* 执行如下脚本抽取编译依赖

    ```sh
    cd tdl_sdk
    #会将依赖抽取到tdl_sdk/dependency/BM1684X文件夹下
    ./scripts/extract_sophon_sdk.sh /path/to/SDK-24.04.01.zip BM1684X
    ```

* 执行编译

    ```shell
    cd tdl_sdk
    ./build_tdl_sdk.sh BM1684X
    ```

### BM1684平台编译

* 在[算能官网](https://developer.sophgo.com/site/index/material/88/all.html)下载SDK-24.04.01的压缩包

* 执行如下脚本抽取编译依赖

    ```sh
    cd tdl_sdk
    #会将依赖抽取到tdl_sdk/dependency/BM1684文件夹下
    ./scripts/extract_sophon_sdk.sh /path/to/SDK-24.04.01.zip BM1684
    ```

* 执行编译

    ```shell
    cd tdl_sdk
    ./build_tdl_sdk.sh BM1684
    ```

### CMODEL_CVITEK平台编译

* 执行如下脚本抽取编译依赖

    ```shell
    cd tdl_sdk
    # 会将依赖下载到tdl_sdk/dependency/CMODEL文件夹下
    ./scripts/extract_cvitek_tpu_sdk.sh
    ```

* 执行编译

    ```shell
    cd tdl_sdk
    ./build_tdl_sdk.sh CMODEL
    ```

### BM1688 Python导出包编译

* 到目标盒子上安装Python环境

    ```shell
    sudo apt-get install python3.8 python3-pip
    pip3 install pybind11
    ```

* 如果没有安装nfs客户端，则需要安装

    ```shell
    sudo apt install nfs-common
    ```

* 把host主机上面的sdk_package目录挂载到目标盒子上

    ```shell
    cd /data
    mkdir sdk_package
    #HOST_IP为编译主机IP
    sudo mount -t nfs HOST_IP:/path/to/sdk_package /data/sdk_package
    ```

* 在盒子上收集Python编译依赖

    ```shell
    cd /data/sdk_package/tdl_sdk/
    chmod +x scripts/extract_python_depends.sh
    #sudo chmod 777 -R dependency #假如dependency目录没有写入权限
    ./scripts/extract_python_depends.sh BM1688
    ```

* 到X86主机上重新执行编译

    ```shell
    cd /data/sdk_package/tdl_sdk/
    ./build_tdl_sdk.sh BM1688
    ```

### BM1684X Python导出包编译

* 到目标盒子上安装Python环境

    ```shell
    sudo apt-get install python3.8 python3-pip
    pip3 install pybind11
    ```

* 如果没有安装nfs客户端，则需要安装

    ```shell
    sudo apt install nfs-common
    ```

* 把host主机上面的sdk_package目录挂载到目标盒子上

    ```shell
    cd /data
    mkdir sdk_package
    #HOST_IP为编译主机IP
    sudo mount -t nfs HOST_IP:/path/to/sdk_package /data/sdk_package
    ```

* 在盒子上收集Python编译依赖

    ```shell
    cd /data/sdk_package/tdl_sdk/
    chmod +x scripts/extract_python_depends.sh
    #sudo chmod 777 -R dependency #假如dependency目录没有写入权限
    ./scripts/extract_python_depends.sh BM1684X    
    ```

* 到X86主机上重新执行编译

    ```shell
    cd /data/sdk_package/tdl_sdk/
    ./build_tdl_sdk.sh BM1684X
    ```

### 静态编译sample

* sample的默认编译方式为动态编译，即编译好后的sample需要配置LD_LIBRARY_PATH环境变量才能执行

    ``` shell
    export LD_LIBRARY_PATH=path/to/lib
    ```

* 如果想直接运行sample，我们提供了静态编译方式，请参考如下修改

    ``` cmake
    # CMakeLists.txt

    # default
    set(BUILD_SHARED ON)

    # static build
    set(BUILD_SHARED OFF)
    ```

* 需要注意，静态编译的sample文件大小将增大，请合理决定