#pragma once
#include <string>
#include <vector>
#include "media_analysis_task.hpp"

class IdentityBrowseTask : public MediaAnalysisTask {
 public:
  IdentityBrowseTask() = default;
  virtual ~IdentityBrowseTask() = default;

  virtual std::string get_event_type() const override {
    return "browse_identity";
  }

  virtual json handle_event(const json& request,
                            const std::string& description) override;

 private:
  void parse_identity_info(const std::string& filename, json& item);
};
