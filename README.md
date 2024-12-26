# TDL_SDK 公版深度学习软件包（Turnkey Deep Learning SDK）

该文档介绍如何编译及使用tdl_sdk。

该版本已完成对

- v4.1.0
- v4.2.0
- athena2
- mars3

四个版本的SDK支持，只需将tdl_sdk放置在对应版本的项目根路径中便可使用。


## 下载

在空文件夹中下载cvi_manifest来辅助下载不同版本的SDK：

``` shell
# 内部从gerrit上clone
git clone ssh://user.name@gerrit-ai.sophgo.vip:29418/cvi_manifest
```

根据不同的芯片，先下载对应的项目包：

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

## 编译

首先，在编译tdl_sdk之前，需要确保项目已完成编译，因为tdl_sdk将依赖其中部分库：

``` shell
source build/envsetup_soc.sh

# 请依据对应的板卡进行选择
defconfig cv18xxx_xxx_xxx

export TPU_REL=1

clean_all && build_all
```

完成整包SDK的编译之后，便可编译tdl_sdk：

``` shell
cd tdl_sdk

# 编译 modules
./build_tdl_sdk.sh

# 编译 samples
./build_tdl_sdk.sh sample

# 编译 modules + samples
./build_tdl_sdk.sh all
```
