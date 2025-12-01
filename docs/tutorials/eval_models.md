# 模型评测

借助evaluation/eval_all.cpp，可以实现对所有模型的评测。

## 评测数据集要求

* evaluation推理出来的结果，必须不用经过其他中间处理即可直接用于最终的性能评测
* 储存数据集文件名的txt文件中，使用相对路径。
* 标签与数据集放到同一个大目录下
* 标签文件夹以labels结尾，以便用于分割次级目录（如有）
* 检测类数据含多级目录的，结构如下：

    ``` shell
    .
    ├── images
    │   ├── soda10m_val
    │   ├── val2017
    │   └── videoimg
    ├── img_list.txt
    └── labels
        ├── soda10m_val
        ├── val2017
        └── videoimg
    ```

* 图片宽高尺寸不超过1920*1080，名字不带有括号空格等特殊字符
* 分类任务目录结构如下，每个数字代表一类：

    ``` shell
    .
    ├── 0
    ├── 1
    ├── 2
    ├── 3
    ├── 4
    └── file_list.txt
    ```

* 评测脚本参数顺序统一，如下：

    ``` shell
    std::string model_name = argv[1];
    std::string model_dir = argv[2];
    std::string file_list_path = argv[3];
    std::string image_root = argv[4];
    std::string result_dir = argv[5];
    ```

## 评测方式

* 运行以下脚本

    ``` shell
    ./eval_all \
        model_info_json \          #eval_info.json 路径
        model_dir \                #模型文件夹路径
        dataset_dir \              #数据集相对于eval_info.json中路径信息的上级目录
        output_dir                 #输出文件夹路径
    ```
