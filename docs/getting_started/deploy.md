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
