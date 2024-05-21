# 公版深度学习TDL（turnkey deep learning）

## 如何编译整包SDK在SOC方式

第一次下载整包sdk_package

``` shell
## zhenjie.wu应该替换为开发者名字
git clone ssh://zhenjie.wu@gerrit-ai.sophgo.vip:29418/cvitek/cvi_manifest.git
## 第一次经过漫长等待
./cvi_manifest/cvitek_repo_clone.sh --gitclone cvi_manifest/default.xml
```

编译cv181x或者cv180x
``` shell
## 编译181x板子，建议借板子的时候也应该借cv1811c_wevb_0006a_spinor这个款板子，保证环境是一样的
## 同理，编译180x板子也应该借cv1801c_wevb_0009a_spinor这个款板子
source build/envsetup_soc.sh
defconfig cv1811c_wevb_0006a_spinor
clean_all && build_all
```

### 内部Release的编译方式
``` shell
build_ai_sdk
```

### 对外开源的TDL_SDK编译方式
``` shell
build_ai_sdk Release
````