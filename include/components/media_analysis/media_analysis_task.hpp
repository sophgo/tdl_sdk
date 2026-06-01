#pragma once
#include <json.hpp>
#include <string>
#include <vector>
#include "utils/tdl_log.hpp"

using json = nlohmann::json;

class MediaAnalysisTask {
 public:
  virtual ~MediaAnalysisTask() = default;
  virtual std::string get_event_type() const = 0;

  // Returns additional event types this task handles (besides get_event_type())
  virtual std::vector<std::string> get_extra_event_types() const { return {}; }

  virtual json handle_event(const json& request,
                            const std::string& description) = 0;
};
