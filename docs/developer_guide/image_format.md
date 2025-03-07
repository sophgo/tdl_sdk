# 图像格式

## YUV 格式

YUV三通道的占比分为如下几种：

* YUV 420，由 4 个 Y 分量共用一套 UV 分量，
* YUV 422，由 2 个 Y 分量共用一套 UV 分量
* YUV 444，不共用，一个 Y 分量使用一套 UV 分量

YUV 的存储方式分为如下几种：

* 平面存储(Planar)，YUV 三个分量分开存储
* 半平面存储(Semi-Planar)，Y作为一个平面，UV分量作为一个平面交叉存储
* 打包存储(Packed)，YUV 三个分量交叉存储

### YUV420SP_UV

* 别名为NV12
* YUV420SP_UV 是 YUV 420 的半平面存储方式，Y 分量作为一个平面，UV 分量作为一个平面交叉存储。
* Y平面的数据量为 width × height，UV平面的数据量为 width × height / 2。
* 总数据量为 width × height × 3 / 2。
* 排布格式如下

    ```sh
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    U V U V U V
    U V U V U V
    U V U V U V
    ```

### YUV420SP_VU

* 别名为NV21
* YUV420SP_VU 是 YUV 420 的半平面存储方式，Y 分量作为一个平面，VU 分量作为一个平面交叉存储。
* Y平面的数据量为 width × height，UV平面的数据量为 width × height / 2。
* 总数据量为 width × height × 3 / 2。
* 排布格式如下

    ```sh
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    V U V U V U
    V U V U V U
    V U V U V U
    ```

### YUV420P_UV

* 别名为I420
* YUV420P_UV 是 YUV 420 的平面存储方式，YUV 三个分量分开存储。
* Y平面的数据量为 width × height，U平面的数据量为 width/2 × height/2，V平面的数据量为 width/2 × height/2。
* 总数据量为 width × height × 3 / 2。
* 排布格式如下

    ```sh
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    U U U
    U U U
    U U U
    V V V
    V V V
    V V V
    ```

### YUV420P_VU

* 别名为YV12
* YUV420P_VU 是 YUV 420 的平面存储方式，YVU 三个分量分开存储。
* Y平面的数据量为 width × height，V平面的数据量为 width/2 × height/2，U平面的数据量为 width/2 × height/2。
* 总数据量为 width × height × 3 / 2。
* 排布格式如下

    ```sh
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    V V V
    V V V
    V V V
    U U U
    U U U
    U U U
    ```

### YUV422P_UV

* 别名为I422
* YUV422P_UV 是 YUV 422 的平面存储方式，YUV 三个分量分开存储。
* Y平面的数据量为 width × height，U平面的数据量为 width × height/2，V平面的数据量为 width × height/2。
* 总数据量为 width × height × 2。
* 排布格式如下

    ```sh
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    U U U U U U
    U U U U U U
    U U U U U U
    V V V V V V
    V V V V V V
    V V V V V V
    ```

### YUV422P_VU

* 别名为YV16
* YUV422P_VU 是 YUV 422 的平面存储方式，YUV 三个分量分开存储，与 YUV422P_UV 的不同是先V后U。
* Y平面的数据量为 width × height，V平面的数据量为 width × height/2，U平面的数据量为 width × height/2。
* 总数据量为 width × height × 2。
* 排布格式如下

    ```sh
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    V V V V V V
    V V V V V V
    V V V V V V
    U U U U U U
    U U U U U U
    U U U U U U
    ```

### YUV422SP_UV

* 别名为NV16
* YUV422SP_UV 是 YUV 422 的半平面存储方式，Y 分量作为一个平面，UV 分量作为一个平面交叉存储。
* Y平面的数据量为 width × height，UV平面的数据量为 width × height。
* 总数据量为 width × height × 2。
* 排布格式如下

    ```sh
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    U V U V U V
    U V U V U V
    U V U V U V
    U V U V U V
    U V U V U V
    U V U V U V
    ```

### YUV422SP_VU

* 别名为NV21
* YUV422SP_VU 是 YUV 422 的半平面存储方式，Y 分量作为一个平面，VU 分量作为一个平面交叉存储。
* Y平面的数据量为 width × height，UV平面的数据量为 width × height。
* 总数据量为 width × height × 2。
* 排布格式如下

    ```sh
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    V U V U V U
    V U V U V U
    V U V U V U
    V U V U V U
    V U V U V U
    V U V U V U
    ```

### YUV444P_UV

* 别名为I444
* YUV444P_UV 是 YUV 444 的平面存储方式，YUV 三个分量分开存储。
* Y平面的数据量为 width × height，U平面的数据量为 width × height，V平面的数据量为 width × height。
* 总数据量为 width × height × 3。
* 排布格式如下

    ```sh
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    U U U U U U
    U U U U U U
    U U U U U U
    U U U U U U
    U U U U U U
    U U U U U U
    V V V V V V
    V V V V V V
    V V V V V V
    V V V V V V
    V V V V V V
    V V V V V V
    ```

### YUV444P_VU

* 别名为YV24
* YUV444P_VU 是 YUV 444 的平面存储方式，YUV 三个分量分开存储，与 YUV444P_UV 的不同是先V后U。
* Y平面的数据量为 width × height，U平面的数据量为 width × height，V平面的数据量为 width × height。
* 总数据量为 width × height × 3。
* 排布格式如下

    ```sh
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    Y Y Y Y Y Y
    V V V V V V
    V V V V V V
    V V V V V V
    V V V V V V
    V V V V V V
    V V V V V V
    U U U U U U
    U U U U U U
    U U U U U U
    U U U U U U
    U U U U U U
    ```

### 单通道图像
