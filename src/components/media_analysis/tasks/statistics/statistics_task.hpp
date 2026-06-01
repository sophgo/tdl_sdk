#pragma once

#include <memory>
#include <string>
#include <vector>
#include "components/media_analysis/media_analysis_task.hpp"

class StatisticsTask : public MediaAnalysisTask {
 public:
  StatisticsTask();
  virtual ~StatisticsTask() = default;

  std::string get_event_type() const override { return "statistics"; }

  std::vector<std::string> get_extra_event_types() const override {
    return {"persons.list",          "person.detail",       "person.timeline",
            "persons.search.names",  "persons.search.face", "dashboard.stats",
            "images.semantic_search"};
  }

  json handle_event(const json& request,
                    const std::string& description) override;

 private:
  json handlePersonsList(const json& request);
  json handlePersonDetail(const json& request);
  json handlePersonTimeline(const json& request);
  json handleSearchNames(const json& request);
  json handleSearchFace(const json& request);
  json handleDashboardStats(const json& request);
  json handleImagesSemanticSearch(const json& request);

  json buildResponse(const json& request, const std::string& event);
};