# 部署指南

## BM1688部署

### 拷贝tld_sdk/install/BM1688

- 创建安装目录

```shell
mkdir /data/tdl_sdk
```

- 拷贝tld_sdk/install/BM1688到安装目录

```shell
cd tdl_sdk
cp -r install/BM1688 /data/tdl_sdk/
```

### 执行以下脚本配置环境变量

```shell
source /data/tdl_sdk/BM1688/scripts/envsetup.sh BM1688 DEPLOY
```