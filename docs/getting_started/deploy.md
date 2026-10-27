# 部署指南

## BM1688部署

- 拷贝编译端的tld_sdk/install/BM1688

```shell
# 在编译端执行以下拷贝命令
cd tdl_sdk
scp -r install/BM1688 linaro@<BM1688_IP>:/data/tdl_sdk/
```

- 在BM1688设备中配置环境

```shell
source /data/tdl_sdk/BM1688/scripts/envsetup.sh BM1688 DEPLOY
```

- 运行示例

```shell
cd tdl_sdk/install/BM1684X/bin
./sample_img_fd /data/sdk_package/tdl_models/bm1684x/ /path/to/xx.jpg
```

## BM1684X部署

- 拷贝编译端的tld_sdk/install/BM1684X

```shell
# 在编译端执行以下拷贝命令
cd tdl_sdk
scp -r install/BM1684X linaro@<BM1684X_IP>:/data/tdl_sdk/
```

- 在BM1684X设备中配置环境

```shell
source /data/tdl_sdk/BM1684X/scripts/envsetup.sh BM1684X DEPLOY
```

- 运行示例

```shell
cd tdl_sdk/install/BM1684X/bin
./sample_img_fd /data/sdk_package/tdl_models/bm1684x/ /path/to/xx.jpg
```

## BM1684部署

- 拷贝编译端的tld_sdk/install/BM1684

```shell
# 在编译端执行以下拷贝命令
cd tdl_sdk
scp -r install/BM1684 linaro@<BM1684_IP>:/data/tdl_sdk/
```

- 在BM1684设备中配置环境

```shell
source /data/tdl_sdk/BM1684/scripts/envsetup.sh BM1684 DEPLOY
```

- 运行示例

```shell
cd tdl_sdk/install/BM1684/bin
./sample_img_fd /data/sdk_package/tdl_models/bm1684/ /path/to/xx.jpg
```

## CV190X部署

CV190X/CV1900B的TDL应用从SD卡或NFS部署，不放入SPI-NOR rootfs。请使用官方打包脚本生成并校验安装包：

```shell
cd tdl_sdk
./scripts/package_cv190x.sh install/CV190X install/CV190X.tar.gz
```

脚本会保留OpenCV和zlib的SONAME符号链接。不要使用`tar -h`或`--dereference`打包；否则可能把符号链接转换为硬链接，导致板端无法找到`libopencv_*.so.4.5`和`libz.so.1`。

传到板端并运行样例：

```shell
# 板端
cd /mnt/sd/tdl_sdk
tar -xzf CV190X.tar.gz
cd CV190X
source scripts/sample_envsetup.sh
./scripts/verify_cv190x_package.sh .
./bin/c/sample_object_detection \
 -m ../models/yolov8n_det_person_vehicle_384_640_INT8_cv190x.cvimodel \
 -i ../dataset/val2017/000000001000.jpg
```

如果通过普通文件复制而不是归档传输，请确认以下名称均存在且可解析：

```text
sample/3rd/opencv/lib/libopencv_core.so.4.5
sample/3rd/opencv/lib/libopencv_imgproc.so.4.5
sample/3rd/opencv/lib/libopencv_imgcodecs.so.4.5
sample/3rd/zlib/lib/libz.so.1
```
