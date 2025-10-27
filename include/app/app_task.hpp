#ifndef APP_TASK_HPP
#define APP_TASK_HPP
#include <json.hpp>
#include "pipeline/pipeline_channel.hpp"
#include "pipeline/pipeline_data_types.hpp"
enum class APP_TASK_TYPE {
  FACE_CAPTURE,
  PERSON_COUNTING,
  PERSON_INTRUSION,
  PERSON_FALL_DOWN,
  VEHICLE_ADAS
};

class AppTask {
 public:
  /**
   * @brief Construct a new App Task object
   *
   * @param task_name
   * @param json_config
   */
  AppTask(const std::string &task_name, const std::string &json_config_file);
  ~AppTask();
  std::vector<std::string> getChannelNames();
  std::string getChannelNodeName(const std::string &channel_name, size_t index);
  int getProcessingChannelNum();
  int getChannelMaxProcessingNum(const std::string &channel_name);
  bool isExternalFrameChannel(const std::string &channel_name);
  int32_t removeChannel(const std::string &channel_name);
  int32_t setFrame(const std::string &pipeline_name,
                   std::shared_ptr<BaseImage> image, uint64_t frame_id);

  virtual int32_t init() = 0;
  virtual int32_t release() = 0;

  virtual int32_t getResult(const std::string &pipeline_name,
                            Packet &result) = 0;
  int send_interval = 1;

 protected:
  std::string task_name_;
  // one task could contain multiple pipeline channels
  std::map<std::string, std::shared_ptr<PipelineChannel>> pipeline_channels_;
  nlohmann::json json_config_;
};

class AppFactory {
 public:
  static std::shared_ptr<AppTask> createAppTask(
      const std::string &task_name, const std::string &json_config_file);
};

#endif
