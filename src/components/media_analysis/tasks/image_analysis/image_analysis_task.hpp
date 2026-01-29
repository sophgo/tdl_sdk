#pragma once
#include <memory>
#include <string>
#include "components/media_analysis/media_analysis_task.hpp"

class ImageAnalysisTask : public MediaAnalysisTask {
 public:
  ImageAnalysisTask(const std::string& data_path);
  virtual ~ImageAnalysisTask() = default;

  std::string get_event_type() const override { return "image_analysis"; }
  json handle_event(const json& request,
                    const std::string& description) override;

  // Additional method for cyclic execution
  json run_analysis_step();

 private:
  std::string data_path_;
  std::string time_str_;
  int counter_ = 0;
};
