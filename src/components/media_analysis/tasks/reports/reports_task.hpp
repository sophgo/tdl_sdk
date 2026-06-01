#pragma once

#include <memory>
#include <string>
#include <vector>
#include "components/media_analysis/media_analysis_task.hpp"

class ReportsTask : public MediaAnalysisTask {
 public:
  ReportsTask();
  virtual ~ReportsTask() = default;

  std::string get_event_type() const override { return "reports"; }

  std::vector<std::string> get_extra_event_types() const override {
    return {"reports.list", "report.generate"};
  }

  json handle_event(const json& request,
                    const std::string& description) override;

  // Called externally to generate a daily report
  json generateDailyReport(const std::string& date_key);
  json generateWeeklyReport(const std::string& week_start_date);

 private:
  json handleReportsList(const json& request);
  json handleReportGenerate(const json& request);

  json buildResponse(const json& request, const std::string& event);

  std::string buildDailyReportPrompt(const std::string& date_key,
                                     const json& all_stats);
  void sendReportToCloud(const json& report);
};