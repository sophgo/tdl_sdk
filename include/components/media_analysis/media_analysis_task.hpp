#pragma once
#include <json.hpp>
#include <string>
#include "utils/tdl_log.hpp"

using json = nlohmann::json;

class MediaAnalysisTask {
 public:
  virtual ~MediaAnalysisTask() = default;
  virtual std::string get_event_type() const = 0;
  virtual json handle_event(const json& request,
                            const std::string& description) = 0;
};
