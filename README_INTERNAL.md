# **TDL_SDK 公版深度学习软件包（Turnkey Deep Learning SDK）**

该文档介绍如何编译及使用TDL_SDK。

该版本已完成对以下四个版本的SDK支持：

- v4.1.0
- v4.2.0
- athena2
- mars3

只需将tdl_sdk放置在对应版本的项目根路径中便可使用。

## 下载

在空文件夹中下载 `cvi_manifest` 来辅助下载不同版本的SDK：

``` shell
# 内部从gerrit上clone cvi_manifest仓库
git clone ssh://gerrit-ai.sophgo.vip:29418/cvi_manifest
```

根据不同的芯片型号及场景，先下载对应的项目包：

``` shell
# cv181x/cv180x v4.1.0
./cvi_manifest/cvitek_repo_clone.sh --gitclone cvi_manifest/golden/cv181x_cv180x_v4.1.0.xml

# cv181x/cv180x v4.2.0
./cvi_manifest/cvitek_repo_clone.sh --gitclone cvi_manifest/golden/cv181x_cv180x_v4.2.0.xml

# bm1688/cv186ah
./cvi_manifest/cvitek_repo_clone.sh --gitclone cvi_manifest/athena2/master/all_repos.xml

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

# cv1811h spinor v4.1.0/v4.2.0
defconfig cv1811h_wevb_0007a_spinor
# athena2 device sdk
defconfig device_wevb_emmc

export TPU_REL=1

# cv181x/cv180x
clean_all && build_all
# edge sdk
clean_edge_all && build_edge_all
# device sdk
clean_device_all && build_device_all
```

`build_all` 命令中已经包含了tdl_sdk的编译，后续再次编译时无需再执行 `build_all` ，可以直接执行 `build_tdl_sdk` ：

``` shell
# 编译TDL_SDK命令
build_tdl_sdk
```

完成整包SDK的编译之后，编译tdl_sdk也可以如下执行：

``` shell
cd tdl_sdk

# 编译 modules
./build_tdl_sdk.sh

# 编译 samples
./build_tdl_sdk.sh sample

# 编译 modules + samples，等同于build_tdl_sdk
./build_tdl_sdk.sh all
```

脚本提供 `clean` 选项来删除已下载的第三方库及其编译产物，但在同样配置下，
重复编译无需使用 `clean` 选项，直接执行编译即可：

``` shell
# 清理所下载的第三方库及其编译产物
./build_tdl_sdk.sh clean
```

---

## 独立编译TDL_SDK

除了依赖于整包SDK进行释放和编译，TDL_SDK针对内部开发者还提供了单独释放和编译的方式。

以下将介绍开发者如何在TDL_SDK中进行快速开发。

### 下载TDL_SDK仓库

TDL_SDK仓库的下载可以通过 `cvi_manifest` 仓库辅助下载，也可以直接下载TDL_SDK仓库。

``` shell
# 下载cvi_manifest
git clone ssh://gerrit-ai.sophgo.vip:29418/cvi_manifest

# 通过cvi_manifest辅助下载tdl_sdk
./cvi_manifest/cvitek_repo_clone.sh --gitclone cvi_manifest/tdl_sdk.xml
```

``` shell
# 直接从gerrit上下载tdl_sdk仓库
git clone ssh://gerrit-ai.sophgo.vip:29418/cvitek/tdl_sdk
```

### 环境变量设置

在开始编译之前，请先添加相应板卡配置的环境变量：

``` shell
# 例如，v4.1.0版本SDK的CV1811H板卡的配置环境变量如下
export CHIP_ARCH=CV181X SDK_VER=musl_riscv64 DUAL_OS=OFF

# 例如，v4.2.0版本SDK的CV1811H板卡的配置环境变量如下
export CHIP_ARCH=CV181X SDK_VER=musl_riscv64 DUAL_OS=ON
```

其中相应环境变量所支持的类型如下，

- CHIP_ARCH: CV180X CV181X CV182X CV183X (default: none)
- SDK_VER: 32bit 64bit glibc_riscv64 musl_riscv64 uclibc (default: none)
- ARM64_VER: v631 v930 v1131 (default: v631)
- DUAL_OS: OFF ON (default: OFF)

以上环境变量中，

`CHIP_ARCH` 、 `SDK_VER` 是必须的；当 `SDK_VER` = 64bit 时， `ARM64_VER` 是必须的。

`ARM64_VER` 中， `v631` 是CV181X板卡使用的； `v1131` 只提供给CV186X的板卡使用，而 `v930` 是旧版本，目前由 `v1131` 替代，请优先使用 `v1131` 版本。

`DUAL_OS` 的启动与否取决于编译出的应用是否需要运行在v4.2.0版本的固件环境中，即v4.1.0选择 `OFF` ，v4.2.0选择 `ON` 。

在添加了相应配置的环境变量后，即可开始进行编译。

根据不同的 `CHIP_ARCH` 和 `BOARD` ，对应着不同的 `SDK_VER` ，其中默认情况使用 `BOARD` 为 `-` 的配置，
其余芯片型号及板卡配置如下表，请根据具体的板卡选择对应的配置。

| CHIP_ARCH | BOARD                        | SDK_VER      |
| :-------- | :--------------------------- | :----------- |
| CV180X    | -                            | musl_riscv64 |
| CV181X    | -                            | musl_riscv64 |
| CV181X    | cv1810ca_wevb_0006a_spinor   | 32bit        |
| CV181X    | cv1810ha_wevb_0006a_spinor   | 32bit        |
| CV181X    | cv1811ca_wevb_0006a_spinand  | 64bit        |
| CV181X    | cv1812cpa_wevb_0006a_spinand | 64bit        |
| CV181X    | cv1812cpa_wevb_0006a_emmc    | 64bit        |
| CV181X    | cv1812ha_wevb_0007a_spinand  | 64bit        |

### 快速编译

TDL_SDK提供快速编译方式：

``` shell
# 进入TDL_SDK仓库执行编译脚本
cd tdl_sdk
./scripts/quick_build_tdl.sh all
```

通过这种方式，TDL_SDK会根据需要从FTP上下载所依赖的第三方库或软件包，
并且针对不同的工具链将下载不同的第三方库保存在本地，后续编译不会再重复下载。

脚本同样提供clean选项来删除已下载的第三方库及其编译产物：

``` shell
# 清理所下载的第三方库及其编译产物
./scripts/quick_build_tdl.sh clean
```

> **注意！** 更换了芯片型号，或者更换了工具链，必须先使用 `clean` 选项删除原工具链编译出来的产物，
才能再执行 `build` 指令重新编译 TDL_SDK。

# 固件更新

一般情况无需频繁更新固件，TDL_SDK作为应用层软件，进行过修改之后只需将应用放置于板子上运行即可。

只有当应用程序在运行时出现了依赖库无法正常运行时，可考虑重新烧录最新版本的固件。

最新固件获取位置为：

``` shell
# 例如，cv1811h spinor v4.1.0版本的固件路径为
ftp://swftp@10.80.0.5/sw_rls/daily_build/projects/cv181x_cv180x_sg200x_v4.1.0/latest/images/normal/soc_cv1811h_wevb_0007a_spinor/upgrade.zip

# 例如，cv1811h spinor v4.2.0版本的固件路径为
ftp://swftp@10.80.0.5/sw_rls/daily_build/projects/cv181x_cv180x_v4.2.0/latest/images/normboot/soc_cv1811h_wevb_0007a_spinor/upgrade.zip
```

请根据实际芯片型号和板卡型号选择对应的文件夹里的固件压缩包！