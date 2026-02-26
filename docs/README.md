```

docs/
├── README.md                         # 目录结构说明
├── getting_started/                  # 入门指南
│   ├── build.md                      # 构建说明
│   ├── run.md                        # 运行示例
│   └── deploy.md                     # 部署说明
├── api_reference/                    # API参考
│   ├── c/                            # C API 文档
│   └── python/                       # Python API 文档
├── developer_guide/                  # 开发者指南
│   ├── code_structure.md             # 代码结构说明
│   ├── image_format.md               # 支持图像格式
│   ├── framework_design.md           # 框架设计说明
│   ├── yolo_development_guide.md     # YOLO系列接口使用说明
│   └── assets/                       # 文档资源
├── tutorials/                        # 教程
│   ├── run_private_model.md          # 运行私有模型
│   └── integrate_new_model.md        # 集成新模型
├── api_poster/                       # API接口
│   └──api_poster_guide.md            # API接口使用说明
└── images/                           # 文档图片

```

## 设计文档（可选子模块：design_docs）

本仓库通过 git submodule 引入 `docs/design_docs`，用于存放方案设计/落地文档（例如 `media_analysis` 等），避免将大量文档历史直接合入主仓库，同时便于独立维护。

- **按需获取（避免不必要的 clone/更新开销）**：默认 clone 不会拉取该子模块；仅在需要阅读/更新设计文档时执行如下命令拉取：

  ```bash
  cd tdl_sdk
  git submodule update --init docs/design_docs
  ```

- 当需要同步最新文档时，可基于 `.gitmodules` 中配置的 `branch = master` 执行如下命令同步：

  ```bash
  cd tdl_sdk
  git submodule update --remote --merge docs/design_docs
  # 然后在主仓库提交 docs/design_docs 指针更新
  ```
