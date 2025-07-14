# api_poster 使用说明

[TOC]

## 运行步骤

1. **在服务器上完成编译**

2. **进入板端操作**

    ```sh
    # 在运行 demo 前，请先进入 sdk_package 目录并声明环境变量
    cd /mnt/data/dzy/ziyan.deng/sdk_package_cv184x # 请替换为您的 sdk_package 目录
    export LD_LIBRARY_PATH=/mnt/tpu_files/lib:\
    $(pwd)/install/soc_cv1842hp_wevb_0014a_emmc/rootfs/usr/lib:\
    $(pwd)/tdl_sdk/install/CV184X/lib:\
    $(pwd)/tdl_sdk/install/CV184X/sample/3rd/opencv/lib:\
    $(pwd)/install/soc_cv1842hp_wevb_0014a_emmc/rootfs/system/usr/lib/:\
    $(pwd)/install/soc_cv1842hp_wevb_0014a_emmc/rootfs/system/usr/lib/3rd/:\
    $(pwd)/ramdisk/rootfs/public/libz/uclibc/lib:\
    $(pwd)/tdl_sdk/install/CV184X/sample/3rd/curl/lib:\
    $(pwd)/tdl_sdk/install/CV184X/sample/3rd/libwebsockets/lib
    export BMRUNTIME_USING_FIRMWARE=/mnt/tpu_files/lib/libtpu_kernel_module.so
    ```

3. **每次板端重启后需重新设置系统时钟**

    ```sh
    # 请将时间设置为当前时间减去 8 小时
    date --set="2025-06-26 18:30:00"
    ```

4. **运行 sample_api_client**

    ```sh
    cd ./tdl_sdk/install/CV184X/bin/cpp
    # 方式一：通过 JSON 文件传参
    ./sample_api_client api_client_name function_name path/to/params.json
    # 方式二：直接输入 JSON 字符串（以图生图为例）
    ./sample_api_client volcengine pictureToPicture '{"ak":"","sk":"","req_key":"img2img_disney_3d_style","sub_req_key":"","image_path":"","output_path":""}'
    ```

## 支持功能汇总

| api_client_name | function_name      | 功能描述                                   |
|-----------------|-------------------|--------------------------------------------|
| sophnet         | chat              | 文本对话                                   |
|                 | analyzeImage      | 图像分析                                   |
| volcengine      | pictureToPicture  | 实时图生图，根据文本提示生成图像           |
|                 | backgroundChange  | 背景更换（主体保持），根据文本提示更换背景 |
|                 | humanSegment      | 人像分割，抠出人像                         |
|                 | humanAgetrans     | 人像年龄转换（仅支持目标年龄 5 或 70）     |
|                 | stylizeImage      | 图像风格化，支持多种风格                   |
| aliyun          | imgeditor         | 通用图像编辑，支持多种编辑功能             |
| tts             | synthesize        | 流式语音合成                               |
| asr             | recognize         | 流式语音识别                               |

## 使用说明

### sophnet-chat

**JSON 示例：**

```json
{
    "api_key": "your_api_key",
    "text": "苹果有哪些功效（文本交流）"
}
```

### sophnet-analyzeImage

**JSON 示例：**

```json
{
    "api_key": "your_api_key",
    "text": "请精确识别图片中的主要物品，并以JSON格式返回结果。要求如下:1.优先返回高置信度的物品(忽略模糊或背景元素)。2.只返回识别到的物品的类别名，不需要其他信息(分析图片的提示文本)",
    "image_path": "path/to/picture.jpg"
}
```

### volcengine-pictureToPicture

**JSON 示例：**

```json
{
    "ak": "your access key",
    "sk": "your secret key",
    "ref_prompt": "漫画风（图生图提示词）",
    "image_path": "path/to/picture.jpg",
    "output_path": "path/to/output_picture.jpg"
}
```

### volcengine-backgroundChange

**JSON 示例：**

```json
{
    "ak": "your access key",
    "sk": "your secret key",
    "ref_prompt": "山脉与晚霞（背景提示词）",
    "image_path": "path/to/picture.jpg",
    "output_path": "path/to/output_picture.jpg"
}
```

### volcengine-humanSegment

**JSON 示例：**

```json
{
    "ak": "your access key",
    "sk": "your secret key",
    "image_path": "path/to/picture.jpg（人像图片）",
    "output_path": "path/to/output_picture.jpg"
}
```

### volcengine-humanAgetrans

**JSON 示例：**

```json
{
    "ak": "your access key",
    "sk": "your secret key",
    "target_age": "5（或 70）",
    "image_path": "path/to/picture.jpg",
    "output_path": "path/to/output_picture.jpg"
}
```

### volcengine-stylizeImage

**JSON 示例：**

```json
{
    "ak": "your access key",
    "sk": "your secret key",
    "req_key": "img2img_disney_3d_style",
    "sub_req_key": "",
    "image_path": "path/to/picture.jpg",
    "output_path": "path/to/output_picture.jpg"
}
```

**参数说明：**
  
```text
req_key:
    网红日漫风：img2img_ghibli_style
    3D风: img2img_disney_3d_style
    写实风：img2img_real_mix_style
    天使风：img2img_pastel_boys_style
    动漫风：img2img_cartoon_style
    日漫风：img2img_makoto_style
    公主风：img2img_rev_animated_style
    梦幻风：img2img_blueline_style
    水墨风：img2img_water_ink_style
    新莫奈花园: i2i_ai_create_monet
    水彩风：img2img_water_paint_style
    莫奈花园：img2img_comic_style
    精致美漫：img2img_comic_style
    赛博机械：img2img_comic_style
    精致韩漫：img2img_exquisite_style
    国风-水墨：img2img_pretty_style
    浪漫光影：img2img_pretty_style
    陶瓷娃娃：img2img_ceramics_style
    中国红：img2img_chinese_style
    丑萌粘土：img2img_clay_style
    可爱玩偶：img2img_clay_style
    3D-游戏_Z时代：img2img_3d_style
    动画电影：img2img_3d_style
    玩偶：img2img_3d_style

sub_req_key:
    当req_key为下列风格时，需要传入sub_req_key
    莫奈花园：img2img_comic_style_monet
    精致美漫：img2img_comic_style_marvel
    赛博机械：img2img_comic_style_future
    国风-水墨：img2img_pretty_style_ink
    浪漫光影：img2img_pretty_style_light
    丑萌粘土：img2img_clay_style_3d
    可爱玩偶：img2img_clay_style_bubble
    3D-游戏_Z时代：img2img_3d_style_era
    动画电影：img2img_3d_style_movie
    玩偶：img2img_3d_style_doll
```

### aliyun-imgeditor

**JSON 示例：**

```json
{
    "api_key": "your api key",
    "function": "",
    "ref_prompt": "",
    "image_path": "path/to/picture.jpg",
    "output_path": "path/to/output_picture.jpg"
}
```

**参数说明：**

- function可以选下列参数，不同的function对应不同的功能:
  - stylization_all：全局风格化，当前支持2种风格。[风格和提示词技巧](https://help.aliyun.com/zh/model-studio/wanx-image-edit?spm=a2c4g.11186623.0.0.274f2392lDWRbx#3be4a1e9569kk)
  - stylization_local：局部风格化，当前支持8种风格。[风格和提示词技巧](https://help.aliyun.com/zh/model-studio/wanx-image-edit?spm=a2c4g.11186623.0.0.274f733f6m62vO#9b8864717b5al)
  - description_edit：指令编辑。通过指令即可编辑图像，简单编辑任务优先推荐这种方式。[提示词技巧](https://help.aliyun.com/zh/model-studio/wanx-image-edit?spm=a2c4g.11186623.0.0.3ed4733fsLdBRT#0c932e6efebf7)
  - remove_watermark：去文字水印。[提示词技巧](https://help.aliyun.com/zh/model-studio/wanx-image-edit?spm=a2c4g.11186623.0.0.3ed4733fYZqkrn#c82e609a4f0bq)
  - expand：扩图。[提示词技巧](https://help.aliyun.com/zh/model-studio/wanx-image-edit?spm=a2c4g.11186623.0.0.3ed4733ffTVYVH#4bd67e438bnuv)
  - super_resolution：图像超分。[提示词技巧](https://help.aliyun.com/zh/model-studio/wanx-image-edit?spm=a2c4g.11186623.0.0.3ed4733fDenRJo#b438794ec2agn)
  - colorization：图像上色。[提示词技巧](https://help.aliyun.com/zh/model-studio/wanx-image-edit?spm=a2c4g.11186623.0.0.3ed4733f8JEUur#ade6bb5d8d28j)
  - doodle：线稿生图。[提示词技巧](https://help.aliyun.com/zh/model-studio/wanx-image-edit?spm=a2c4g.11186623.0.0.3ed4733ftCiQhc#b78933f9e819x)
  - control_cartoon_feature：参考卡通形象生图。[提示词技巧](<https://help.aliyun.com/zh/model-studio/wanx-image-edit?spm=a2c4g.11186623.0.0.3ed4733fjZ1rUf#ee9063aeaeagf>

### tts-synthesize

**JSON 示例：**

```json
{
    "appid": "your api key",
    "token": "your token",
    "voice_type": "BV700_V2_streaming",
    "encoding": "pcm",
    "text": "要合成语音的文本",
    "output_path": "path/to/audio.raw（后缀需与 encoding 保持一致）",
    "operation": "submit"
}
```

**参数说明：**

- `voice_type`：音色选择，默认为 BV700_V2_streaming，[音色参考](https://www.volcengine.com/docs/6561/97465)
- `encoding`：编码方式，支持 wav、pcm、ogg_opus、mp3
- `operation`：默认为 submit（流式输出），可选 query（非流式输出）

### asr-recognize

**JSON 示例：**

```json
{
    "appid": "your api key",
    "token": "your token",
    "cluster": "volcengine_streaming_common",
    "audio_path": "path/to/audio.pcm（后缀需与 format 保持一致）",
    "audio_format": "raw"
}
```
