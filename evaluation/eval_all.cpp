#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "json.hpp"
#include "utils/tdl_log.hpp"

using json = nlohmann::json;
using namespace std;

std::string platform;
std::string model_extension;

void getModelExtension(std::string& platform, std::string& model_extension) {
#if defined(__CV180X__)
  platform = "cv180x";
  model_extension = "_cv180x.cvimodel";

#elif defined(__CV181X__) || defined(__CMODEL_CV181X__)
  platform = "cv181x";
  model_extension = "_cv181x.cvimodel";

#elif defined(__CV184X__) || defined(__CMODEL_CV184X__)
  platform = "cv184x";
  model_extension = "_cv184x.bmodel";

#elif defined(__CV186X__)
  platform = "cv186x";
  model_extension = "_cv186x.bmodel";

#elif defined(__BM1688__)
  platform = "bm1688";
  model_extension = "_bm1688.bmodel";

#else
  LOGE("platform not supported");
  assert(false);
#endif
}

// 创建目录的辅助函数，如果路径不存在则递归创建
bool createDirectory(const std::string& path) {
  std::string cmd = "mkdir -p " + path;
  int ret = system(cmd.c_str());
  if (ret != 0) {
    LOGE("Failed to create directory: %s", path.c_str());
    return false;
  }
  return true;
}

// 通用解析函数
// 修改函数参数，增加 model_name
vector<string> parse_common(const json& j, const string& model_dir,
                            const string& dataset_dir, const string& output_dir,
                            const string& key, const string& model_name) {
  vector<string> commands;

  for (const auto& item : j) {
    for (const auto& element : item.items()) {
      const string& current_model_name = element.key();  // 重命名避免变量冲突
      const auto& model_info = element.value();

      // 如果指定了模型名称，只处理匹配的模型
      if (!model_name.empty() && current_model_name != model_name) {
        continue;
      }

      string model_id = "";
      string model_path = model_dir;

      // 检查是否需要按target_chips过滤
      if (model_info.contains("target_chips")) {
        auto target_chips =
            model_info["target_chips"].get<std::vector<std::string>>();
        // 如果当前平台不在目标芯片列表中，则跳过
        if (std::find(target_chips.begin(), target_chips.end(), platform) ==
            target_chips.end()) {
          continue;
        }
      }

      if (model_info.contains("model_id")) {
        model_id = model_info["model_id"];

        model_path = model_dir + "/" + platform + "/" + current_model_name +
                     model_extension;
      }

      // 构建输出路径时使用 current_model_name
      string output = output_dir + "/" + current_model_name + "/";
      createDirectory(output);

      if (key == "eval_classification" || key == "eval_bin_audio" ||
          key == "eval_hand_keypoint_cls" || key == "eval_face_attribute" ||
          key == "eval_license_plate_recognition") {
        output = output + "/" + current_model_name +
                 ".txt";  // 使用 current_model_name
      }

      string file_list =
          dataset_dir + "/" + model_info["file_list"].get<std::string>();
      string real_dataset_dir = dataset_dir + "/" +
                                model_info["dataset_dir"].get<std::string>() +
                                "/";

      // 构建命令
      string cmd = "./" + key + " " + model_id + " " + model_path + " " +
                   file_list + " " + real_dataset_dir + " " + output;
      commands.push_back(cmd);
    }
  }

  return commands;
}

// CLIP pipeline解析函数（特殊结构）
// 修改函数参数，增加 model_name
vector<string> parse_clip_pipeline(const json& j, const string& model_dir,
                                   const string& dataset_dir,
                                   const string& output_dir, const string& key,
                                   const string& model_name) {
  vector<string> commands;

  for (const auto& item : j) {
    for (const auto& element : item.items()) {
      const string& current_model_name = element.key();  // 重命名避免变量冲突
      const auto& model_info = element.value();

      // 如果指定了模型名称，只处理匹配的模型
      if (!model_name.empty() && current_model_name != model_name) {
        continue;
      }

      string model_id_img = model_info["model_id_img"];
      string model_id_text = model_info["model_id_text"];

      string model_name_img = model_info["model_name_img"];
      string model_name_text = model_info["model_name_text"];

      // 检查是否需要按target_chips过滤
      if (model_info.contains("target_chips")) {
        auto target_chips =
            model_info["target_chips"].get<std::vector<std::string>>();
        // 如果当前平台不在目标芯片列表中，则跳过
        if (std::find(target_chips.begin(), target_chips.end(), platform) ==
            target_chips.end()) {
          continue;
        }
      }

      string model_path_img =
          model_dir + "/" + platform + "/" + model_name_img + model_extension;
      string model_path_text =
          model_dir + "/" + platform + "/" + model_name_text + model_extension;

      string file_list =
          dataset_dir + "/" + model_info["file_list"].get<std::string>();
      string real_dataset_dir = dataset_dir + "/" +
                                model_info["dataset_dir"].get<std::string>() +
                                "/";
      string txt_dir =
          dataset_dir + "/" + model_info["txt_dir"].get<std::string>();

      // 构建输出路径时使用 current_model_name
      string output = output_dir + "/" + current_model_name;
      createDirectory(output);

      string cmd = "./" + key + " " + model_id_img + " " + model_id_text + " " +
                   model_path_img + " " + model_path_text + " " +
                   real_dataset_dir + " " + file_list + " " + txt_dir + " " +
                   output + "/" + current_model_name +
                   ".txt";  // 使用 current_model_name
      commands.push_back(cmd);
    }
  }

  return commands;
}

// 音频ASR解析函数
// 修改函数参数，增加 model_name
vector<string> parse_audio_asr(const json& j, const string& model_dir,
                               const string& dataset_dir,
                               const string& output_dir, const string& key,
                               const string& model_name) {
  vector<string> commands;

  for (const auto& item : j) {
    for (const auto& element : item.items()) {
      const string& current_model_name = element.key();  // 重命名避免变量冲突
      const auto& model_info = element.value();

      // 如果指定了模型名称，只处理匹配的模型
      if (!model_name.empty() && current_model_name != model_name) {
        continue;
      }

      // 检查是否需要按target_chips过滤
      if (model_info.contains("target_chips")) {
        auto target_chips =
            model_info["target_chips"].get<std::vector<std::string>>();
        // 如果当前平台不在目标芯片列表中，则跳过
        if (std::find(target_chips.begin(), target_chips.end(), platform) ==
            target_chips.end()) {
          continue;
        }
      }

      string tokens_file =
          dataset_dir + "/" + model_info["tokens_file"].get<std::string>();
      string real_dataset_dir = dataset_dir + "/" +
                                model_info["dataset_dir"].get<std::string>() +
                                "/";
      string file_list =
          dataset_dir + "/" + model_info["file_list"].get<std::string>();

      // 构建输出路径时使用 current_model_name
      string output = output_dir + "/" + current_model_name;
      createDirectory(output);

      string cmd = "./" + key + " " + model_dir + " " + tokens_file + " " +
                   real_dataset_dir + " " + file_list + " " + output + "/" +
                   current_model_name + ".txt";  // 使用 current_model_name
      commands.push_back(cmd);
    }
  }

  return commands;
}

// SOT解析函数（新增key参数并更新命令）
// 修改函数参数，增加 model_name
vector<string> parse_sot(const json& j, const string& model_dir,
                         const string& dataset_dir, const string& output_dir,
                         const string& key, const string& model_name) {
  vector<string> commands;

  for (const auto& item : j) {
    for (const auto& element : item.items()) {
      const string& current_model_name = element.key();  // 重命名避免变量冲突
      const auto& model_info = element.value();

      // 如果指定了模型名称，只处理匹配的模型
      if (!model_name.empty() && current_model_name != model_name) {
        continue;
      }

      // 检查是否需要按target_chips过滤
      if (model_info.contains("target_chips")) {
        auto target_chips =
            model_info["target_chips"].get<std::vector<std::string>>();
        // 如果当前平台不在目标芯片列表中，则跳过
        if (std::find(target_chips.begin(), target_chips.end(), platform) ==
            target_chips.end()) {
          continue;
        }
      }
      string model_id = model_info["model_id"];
      string real_dataset_dir = dataset_dir + "/" +
                                model_info["dataset_dir"].get<std::string>() +
                                "/";
      string labels_dir =
          dataset_dir + "/" + model_info["labels_dir"].get<std::string>();

      // 构建输出路径时使用 current_model_name
      string output = output_dir + "/" + current_model_name + "/";
      createDirectory(output);

      string cmd = "./" + key + " " + model_dir + " " + real_dataset_dir + " " +
                   labels_dir + " " + output;
      commands.push_back(cmd);
    }
  }

  return commands;
}

// 人脸特征解析函数
// 修改函数参数，增加 model_name
vector<string> parse_face_feature(const json& j, const string& model_dir,
                                  const string& dataset_dir,
                                  const string& output_dir, const string& key,
                                  const string& model_name) {
  vector<string> commands;

  for (const auto& item : j) {
    for (const auto& element : item.items()) {
      const string& current_model_name = element.key();  // 重命名避免变量冲突
      const auto& model_info = element.value();

      // 如果指定了模型名称，只处理匹配的模型
      if (!model_name.empty() && current_model_name != model_name) {
        continue;
      }

      // 检查是否需要按target_chips过滤
      if (model_info.contains("target_chips")) {
        auto target_chips =
            model_info["target_chips"].get<std::vector<std::string>>();
        // 如果当前平台不在目标芯片列表中，则跳过
        if (std::find(target_chips.begin(), target_chips.end(), platform) ==
            target_chips.end()) {
          continue;
        }
      }

      string model_id = model_info["model_id"];
      string labels_file =
          dataset_dir + "/" + model_info["labels_file"].get<std::string>();

      // 构建输出路径时使用 current_model_name
      string output = output_dir + "/" + current_model_name;
      createDirectory(output);

      string cmd = "./" + key + " " + model_id + " " + model_dir + " " +
                   labels_file + " " + output + "/" + current_model_name +
                   ".txt";  // 使用 current_model_name
      commands.push_back(cmd);
    }
  }

  return commands;
}

// 辅助函数：执行命令并返回结果
int execute_command(const string& cmd) {
  cout << "Executing: " << cmd << endl;
  return system(cmd.c_str());
}

// 主函数
int main(int argc, char* argv[]) {
  // 修改参数检查，支持 5 个（原有）或 6 个（增加 model_name）参数
  if (argc != 5 && argc != 6) {
    cerr
        << "Usage: " << argv[0]
        << " model_info_json model_dir dataset_dir output_dir [model_name]\n"  // 更新帮助信息
        << "see also: docs/tutorials/eval_models.md " << endl;
    return 1;
  }

  string model_info_json = argv[1];
  string model_dir = argv[2];
  string dataset_dir = argv[3];
  string output_dir = argv[4];
  string
      model_name;  // 可选的模型名称参数（不包含如"_cv184x.bmodel"的后缀），传入后只跑该模型

  // 如果提供了第 6 个参数，则解析为模型名称
  if (argc == 6) {
    model_name = argv[5];
  }

  getModelExtension(platform, model_extension);

  // 读取model_info.json
  ifstream f(model_info_json);
  if (!f.is_open()) {
    cerr << "Failed to model_info_json" << endl;
    return 1;
  }

  json j;
  try {
    f >> j;
  } catch (const json::parse_error& e) {
    cerr << "JSON parse error: " << e.what() << endl;
    return 1;
  }

  // 存储所有要执行的命令
  vector<string> commands;

  // {{ 修改开始 }}
  // 定义解析函数指针类型（增加 model_name 参数）
  using ParseFunc =
      vector<string> (*)(const json&, const string&, const string&,
                         const string&, const string&, const string&);

  // 评估任务映射表：JSON键 -> 对应的解析函数
  vector<pair<string, ParseFunc>> eval_handlers = {
      {"eval_face_detection", parse_common},
      {"eval_object_detection", parse_common},
      {"eval_classification", parse_common},
      {"eval_bin_audio", parse_common},
      {"eval_hand_keypoint_cls", parse_common},
      {"eval_face_attribute", parse_common},
      {"eval_license_plate_recognition", parse_common},
      {"eval_keypoint", parse_common},
      {"eval_pose_yolov8", parse_common},
      {"eval_lane_detection", parse_common},
      {"eval_clip_pipeline", parse_clip_pipeline},
      {"eval_audio_asr", parse_audio_asr},
      {"eval_sot", parse_sot},
      {"eval_face_feature", parse_face_feature}};

  // 循环遍历所有评估任务
  for (const auto& handler : eval_handlers) {
    const string& key = handler.first;
    ParseFunc parse_func = handler.second;
    if (j.contains(key)) {
      // 传递 model_name 参数给解析函数
      auto cmds = parse_func(j[key], model_dir, dataset_dir, output_dir, key,
                             model_name);
      commands.insert(commands.end(), cmds.begin(), cmds.end());
    } else {
      cout << "No key found: " << key << endl;
    }
  }
  // {{ 修改结束 }}

  // 执行所有命令
  for (const string& cmd : commands) {
    std::cout << "Executing: " << cmd << std::endl;
    int ret = execute_command(cmd);
    if (ret != 0) {
      cerr << "Command failed with return code: " << ret << endl;
      // 可以选择继续执行或退出
      // return ret;
    }
  }

  return 0;
}
