#pragma once

#include <memory>
#include <string>
#include <vector>
#include "components/media_analysis/media_analysis_task.hpp"

class BehaviorAnalysisTask : public MediaAnalysisTask {
 public:
  BehaviorAnalysisTask();
  virtual ~BehaviorAnalysisTask() = default;

  std::string get_event_type() const override { return "behavior_analysis"; }

  // Returns additional event types this task handles
  std::vector<std::string> get_extra_event_types() const override {
    return {"behavior.analysis.generate", "video.analysis.generate",
            "behavior.analysis.daily_aggregation", "behavior.analysis"};
  }

  json handle_event(const json& request,
                    const std::string& description) override;

  // ---- Called by external code (e.g. run_tdl_thread) ----

  // Called when a new appearance starts
  json onAppearanceStart(const std::string& person_id,
                         const std::string& channel_id, int track_id,
                         int64_t start_time_ms, int first_frame_id);

  // Called when an appearance ends - triggers per_appearance analysis
  json onAppearanceEnd(const std::string& appearance_id, int64_t end_time_ms,
                       int last_frame_id, const std::string& stats_json);

  // Submit a video for behavior analysis (with SQLite persistence)
  json submitBehaviorVideo(const std::string& video_path,
                           const std::string& person_name,
                           const std::string& person_id,
                           const std::string& appearance_id,
                           uint32_t duration_sec);

  // Update analysis status and push progress event
  void updateAnalysisProgress(const std::string& job_id,
                              const std::string& status, int progress_pct);

  // Mark analysis as completed with results
  void completeAnalysis(const std::string& job_id,
                        const std::string& summary_text,
                        const std::string& key_frame_json, float confidence);

  // Mark analysis as failed
  void failAnalysis(const std::string& job_id, const std::string& error_code,
                    const std::string& error_message, bool retryable = false);

  // Daily aggregation - triggered by timer or manual request
  void triggerDailyAggregation(const std::string& date_key = "");

 private:
  json handleGenerateRequest(const json& request);
  json handleDailyAggregationRequest(const json& request);
  json handleVideoAnalysisRequest(const json& request);
  json handleCloudResponse(const json& request);

  void pushEvent(const json& data);

  static std::string buildAnalysisPrompt(const std::string& person_name,
                                         uint32_t duration_sec);
  static std::string buildDailyAggregationPrompt(
      const std::string& person_id, const std::string& person_name,
      const std::string& date_key, const json& per_appearance_summaries);
  static std::string generateJobId(const std::string& prefix);
  static std::string generateAnalysisId(const std::string& prefix);
  static std::string getDateKey(int64_t timestamp_ms = 0);
};