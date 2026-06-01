#include "behavior_analysis_task.hpp"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <sstream>
#include "components/media_analysis/media_analysis_server.hpp"
#include "smart_home_db.hpp"

BehaviorAnalysisTask::BehaviorAnalysisTask() {}

std::string BehaviorAnalysisTask::generateJobId(const std::string& prefix) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<> dis(1000, 9999);

  auto now = std::chrono::system_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch())
                .count();

  char buf[128];
  snprintf(buf, sizeof(buf), "%s_%lld_%04d", prefix.c_str(), (long long)ms,
           dis(gen));
  return std::string(buf);
}

std::string BehaviorAnalysisTask::generateAnalysisId(
    const std::string& prefix) {
  return generateJobId(prefix);
}

std::string BehaviorAnalysisTask::buildAnalysisPrompt(
    const std::string& person_name, uint32_t duration_sec) {
  char buf[1024];
  snprintf(
      buf, sizeof(buf),
      "你是一个智能家居行为分析助手。以下是摄像头拍摄到的一段人员活动视频。\n\n"
      "视频中人物：%s\n持续时长：%u 秒\n\n"
      "请分析此人在该片段中的具体行为，包括但不限于：\n"
      "1. 活动轨迹（从哪到哪，经过了哪些区域）\n"
      "2. 具体行为（走动、停留、坐下、取物、与人交谈等）\n"
      "3. 情绪状态\n"
      "4. 是否有异常行为\n\n"
      "请用简洁的中文输出分析结果，控制在100字以内。",
      person_name.empty() ? "未知" : person_name.c_str(), duration_sec);
  return std::string(buf);
}

std::string BehaviorAnalysisTask::getDateKey(int64_t timestamp_ms) {
  if (timestamp_ms <= 0) {
    timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  }
  time_t t = timestamp_ms / 1000;
  struct tm tm_buf;
  localtime_r(&t, &tm_buf);
  char buf[16];
  strftime(buf, sizeof(buf), "%Y-%m-%d", &tm_buf);
  return std::string(buf);
}

void BehaviorAnalysisTask::pushEvent(const json& data) {
  json msg;
  msg["schema_version"] = "smart_home.ws.v1";
  msg["message_id"] =
      "evt_" + std::to_string(
                   std::chrono::system_clock::now().time_since_epoch().count());
  msg["timestamp_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
  msg["type"] = "event";
  msg["source"] = "c_backend";
  msg["destination"] = "web_client";
  msg["payload"]["event"] = data["event"];
  msg["payload"]["data"] = data["data"];

  MediaAnalysisServer::GetInstance()->send_to_web_client(msg.dump());
}

// ==================== Appearance Lifecycle ====================

json BehaviorAnalysisTask::onAppearanceStart(const std::string& person_id,
                                             const std::string& channel_id,
                                             int track_id,
                                             int64_t start_time_ms,
                                             int first_frame_id) {
  auto* db = SmartHomeDB::GetInstance();

  std::string appearance_id = generateJobId("app");

  json appearance;
  appearance["appearance_id"] = appearance_id;
  appearance["person_id"] = person_id;
  appearance["channel_id"] = channel_id;
  appearance["track_id"] = track_id;
  appearance["start_time_ms"] = start_time_ms;
  appearance["first_frame_id"] = first_frame_id;

  db->insertAppearance(appearance);

  std::cout << "[BehaviorAnalysis] Appearance started: " << appearance_id
            << " person=" << person_id << " track=" << track_id << std::endl;

  json result;
  result["appearance_id"] = appearance_id;
  result["status"] = "started";
  return result;
}

json BehaviorAnalysisTask::onAppearanceEnd(const std::string& appearance_id,
                                           int64_t end_time_ms,
                                           int last_frame_id,
                                           const std::string& stats_json) {
  auto* db = SmartHomeDB::GetInstance();

  // Get the appearance to calculate duration
  json app = db->getAppearance(appearance_id);
  if (app.empty()) {
    std::cerr << "[BehaviorAnalysis] Appearance not found: " << appearance_id
              << std::endl;
    json err;
    err["status"] = "error";
    err["message"] = "Appearance not found";
    return err;
  }

  int64_t start_time_ms = app["start_time_ms"].get<int64_t>();
  int64_t duration_ms = end_time_ms - start_time_ms;

  db->updateAppearanceEnd(appearance_id, end_time_ms, duration_ms,
                          last_frame_id, stats_json);

  std::string person_id = app["person_id"].get<std::string>();

  // Update person's last_seen_ms
  json person = db->getPerson(person_id);
  if (!person.empty()) {
    person["last_seen_ms"] = end_time_ms;
    db->updatePerson(person);
  }

  int64_t duration_sec = duration_ms / 1000;

  std::cout << "[BehaviorAnalysis] Appearance ended: " << appearance_id
            << " duration=" << duration_sec << "s" << std::endl;

  // Skip analysis for very short appearances (< 3s)
  if (duration_sec < 3) {
    std::cout << "[BehaviorAnalysis] Skipping analysis - too short ("
              << duration_sec << "s < 3s)" << std::endl;
    json result;
    result["status"] = "skipped";
    result["reason"] = "too_short";
    return result;
  }

  // Get person name
  std::string person_name = person.value("display_name", "未知");

  // Create per_appearance analysis task
  std::string job_id = generateJobId("job_per_app");
  std::string analysis_id = generateAnalysisId("ba");
  std::string date_key = getDateKey(start_time_ms);

  json analysis;
  analysis["analysis_id"] = analysis_id;
  analysis["job_id"] = job_id;
  analysis["person_id"] = person_id;
  analysis["appearance_id"] = appearance_id;
  analysis["date_key"] = date_key;
  analysis["analysis_type"] = "per_appearance";
  analysis["trigger_type"] = "auto";
  analysis["status"] = "accepted";
  analysis["progress_pct"] = 0;
  analysis["video_url"] = "";  // Will be set when video is available

  db->insertBehaviorAnalysis(analysis);

  // Push ack event to frontend
  json ack;
  ack["event"] = "behavior.analysis.ack";
  ack["data"]["job_id"] = job_id;
  ack["data"]["person_id"] = person_id;
  ack["data"]["appearance_id"] = appearance_id;
  ack["data"]["status"] = "accepted";
  pushEvent(ack);

  // If a video path is known, submit it
  // (the actual video path would come from the video recording module)
  std::string video_path = "";  // external code sets this
  if (!video_path.empty()) {
    submitBehaviorVideo(video_path, person_name, person_id, appearance_id,
                        (uint32_t)duration_sec);
  }

  json result;
  result["status"] = "analysis_created";
  result["job_id"] = job_id;
  result["analysis_id"] = analysis_id;
  result["appearance_id"] = appearance_id;
  return result;
}

// ==================== Video Submission ====================

json BehaviorAnalysisTask::submitBehaviorVideo(const std::string& video_path,
                                               const std::string& person_name,
                                               const std::string& person_id,
                                               const std::string& appearance_id,
                                               uint32_t duration_sec) {
  std::string prompt = buildAnalysisPrompt(person_name, duration_sec);
  std::string job_id = generateJobId("job_per_app");
  std::string analysis_id = generateAnalysisId("ba");
  std::string date_key = getDateKey();

  // Persist to SQLite
  auto* db = SmartHomeDB::GetInstance();
  json analysis;
  analysis["analysis_id"] = analysis_id;
  analysis["job_id"] = job_id;
  analysis["person_id"] = person_id;
  analysis["appearance_id"] = appearance_id;
  analysis["date_key"] = date_key;
  analysis["analysis_type"] = "per_appearance";
  analysis["trigger_type"] = "auto";
  analysis["status"] = "accepted";
  analysis["progress_pct"] = 0;
  analysis["video_url"] = "/api/video_proxy?path=" + video_path;
  db->insertBehaviorAnalysis(analysis);

  // Push ack
  json ack;
  ack["event"] = "behavior.analysis.ack";
  ack["data"]["job_id"] = job_id;
  ack["data"]["person_id"] = person_id;
  ack["data"]["appearance_id"] = appearance_id;
  ack["data"]["status"] = "accepted";
  pushEvent(ack);

  // Push uploading progress
  updateAnalysisProgress(job_id, "uploading", 20);

  // Send to cloud for inference
  json msg;
  msg["event"] = "behavior.analysis";
  msg["video_url"] = "/api/video_proxy?path=" + video_path;
  msg["prompt"] = prompt;
  msg["job_id"] = job_id;
  msg["person_id"] = person_id;
  msg["person_name"] = person_name;
  msg["appearance_id"] = appearance_id;
  msg["duration_sec"] = duration_sec;
  msg["analysis_type"] = "per_appearance";

  std::cout << "[BehaviorAnalysisTask] Submitting video: " << video_path
            << " person=" << person_name << "(" << person_id << ")"
            << " duration=" << duration_sec << "s job_id=" << job_id
            << std::endl;

  MediaAnalysisServer::GetInstance()->send_to_cloud_client(msg.dump());

  // Push inferring progress (optimistically)
  updateAnalysisProgress(job_id, "inferring", 80);

  json result;
  result["status"] = "submitted";
  result["job_id"] = job_id;
  result["analysis_id"] = analysis_id;
  result["message"] = "Video submitted for analysis";
  return result;
}

// ==================== Progress & Completion ====================

void BehaviorAnalysisTask::updateAnalysisProgress(const std::string& job_id,
                                                  const std::string& status,
                                                  int progress_pct) {
  auto* db = SmartHomeDB::GetInstance();
  db->updateBehaviorAnalysisStatus(job_id, status, progress_pct);

  // Determine analysis_type from DB for proper frontend display
  json analysis = db->getBehaviorAnalysis(job_id);
  std::string analysis_type = analysis.value("analysis_type", "per_appearance");

  json evt;
  evt["event"] = "behavior.analysis.progress";
  evt["data"]["job_id"] = job_id;
  evt["data"]["person_id"] = analysis.value("person_id", "");
  evt["data"]["analysis_type"] = analysis_type;
  evt["data"]["status"] = status;
  evt["data"]["progress_pct"] = progress_pct;

  if (!analysis.value("appearance_id", "").empty()) {
    evt["data"]["appearance_id"] = analysis["appearance_id"];
  }

  pushEvent(evt);
}

void BehaviorAnalysisTask::completeAnalysis(const std::string& job_id,
                                            const std::string& summary_text,
                                            const std::string& key_frame_json,
                                            float confidence) {
  auto* db = SmartHomeDB::GetInstance();

  // Set video expiry to 7 days from now
  int64_t expire_at_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count() +
      7LL * 86400000LL;

  db->completeBehaviorAnalysis(job_id, summary_text, key_frame_json, confidence,
                               expire_at_ms);

  json analysis = db->getBehaviorAnalysis(job_id);
  std::string analysis_type = analysis.value("analysis_type", "per_appearance");

  // Push the result
  if (analysis_type == "per_appearance") {
    json evt;
    evt["event"] = "behavior.analysis.ready";
    evt["data"]["job_id"] = job_id;
    evt["data"]["person_id"] = analysis.value("person_id", "");
    evt["data"]["person_name"] = "";
    evt["data"]["appearance_id"] = analysis.value("appearance_id", "");
    evt["data"]["date_key"] = analysis.value("date_key", "");
    evt["data"]["analysis_type"] = "per_appearance";
    evt["data"]["analysis_text"] = summary_text;
    evt["data"]["video_url"] = analysis.value("video_url", "");
    evt["data"]["key_frames"] =
        json::parse(key_frame_json.empty() ? "[]" : key_frame_json);
    evt["data"]["confidence"] = confidence;

    pushEvent(evt);
  }
}

void BehaviorAnalysisTask::failAnalysis(const std::string& job_id,
                                        const std::string& error_code,
                                        const std::string& error_message,
                                        bool retryable) {
  auto* db = SmartHomeDB::GetInstance();
  db->updateBehaviorAnalysisStatus(job_id, "failed", -1, error_code,
                                   error_message);

  json analysis = db->getBehaviorAnalysis(job_id);

  json evt;
  evt["event"] = "behavior.analysis.error";
  evt["data"]["job_id"] = job_id;
  evt["data"]["person_id"] = analysis.value("person_id", "");
  evt["data"]["error_code"] = error_code;
  evt["data"]["error_message"] = error_message;
  evt["data"]["retryable"] = retryable;

  pushEvent(evt);
}

// ==================== Daily Aggregation ====================

std::string BehaviorAnalysisTask::buildDailyAggregationPrompt(
    const std::string& person_id, const std::string& person_name,
    const std::string& date_key, const json& per_appearance_summaries) {
  std::ostringstream prompt;
  prompt << "你是一个智能家居行为分析助手。以下是 " << person_name << " 今天（"
         << date_key << "）的所有活动记录：\n\n";

  int idx = 1;
  for (const auto& a : per_appearance_summaries) {
    prompt << idx++ << ". " << a.value("summary_text", "") << "\n";
  }

  prompt << "\n请基于以上所有片段，生成一份今日行为总结，包括：\n"
         << "1. 今日活动概览（总共出现几次，主要在哪些区域活动）\n"
         << "2. 主要事件时间线\n"
         << "3. 情绪变化趋势\n"
         << "4. 与其他人的互动情况\n"
         << "5. 异常/值得关注的事件\n"
         << "\n请控制在200字以内。";
  return prompt.str();
}

void BehaviorAnalysisTask::triggerDailyAggregation(
    const std::string& date_key_override) {
  std::string date_key =
      date_key_override.empty() ? getDateKey() : date_key_override;
  auto* db = SmartHomeDB::GetInstance();

  std::cout << "[BehaviorAnalysis] Starting daily aggregation for " << date_key
            << std::endl;

  // Get all persons
  json persons_data = db->listPersons("all", "last_seen_ms", "desc", 1, 100);

  for (const auto& person : persons_data["items"]) {
    std::string person_id = person["person_id"].get<std::string>();
    std::string person_name = person.value("display_name", "未知");

    // Get today's per_appearance analyses
    json per_app = db->getPerAppearanceAnalyses(person_id, date_key);

    // Count completed analyses
    int completed_count = 0;
    json completed_analyses = json::array();
    for (const auto& a : per_app["items"]) {
      if (a.value("status", "") == "completed") {
        completed_analyses.push_back(a);
        completed_count++;
      }
    }

    if (completed_count == 0) {
      // No data to aggregate
      continue;
    }

    std::string job_id = generateJobId("job_daily_agg");
    std::string analysis_id = generateAnalysisId("ba_daily");

    // Create daily_aggregation record
    json analysis;
    analysis["analysis_id"] = analysis_id;
    analysis["job_id"] = job_id;
    analysis["person_id"] = person_id;
    analysis["appearance_id"] = nullptr;
    analysis["date_key"] = date_key;
    analysis["analysis_type"] = "daily_aggregation";
    analysis["trigger_type"] = "auto";
    analysis["status"] = "daily_aggregating";
    analysis["progress_pct"] = 0;

    // Collect child analysis IDs for key_frame_json
    json child_ids = json::array();
    for (const auto& a : completed_analyses) {
      child_ids.push_back(a["analysis_id"]);
    }
    analysis["key_frame_json"] = child_ids.dump();

    db->insertBehaviorAnalysis(analysis);

    // Push progress: daily_aggregating
    json prog;
    prog["event"] = "behavior.analysis.progress";
    prog["data"]["job_id"] = job_id;
    prog["data"]["person_id"] = person_id;
    prog["data"]["analysis_type"] = "daily_aggregation";
    prog["data"]["status"] = "daily_aggregating";
    prog["data"]["progress_pct"] = 50;
    pushEvent(prog);

    if (completed_count == 1) {
      // Only 1 appearance - reuse its summary directly, skip LLM
      std::string summary = completed_analyses[0].value("summary_text", "");

      db->completeBehaviorAnalysis(job_id, summary, child_ids.dump(), 0.80f);

      // Push daily_aggregation.ready
      json evt;
      evt["event"] = "behavior.analysis.daily_aggregation.ready";
      evt["data"]["job_id"] = job_id;
      evt["data"]["person_id"] = person_id;
      evt["data"]["person_name"] = person_name;
      evt["data"]["date_key"] = date_key;
      evt["data"]["analysis_type"] = "daily_aggregation";
      evt["data"]["daily_summary"] = summary;
      evt["data"]["appearance_count"] = 1;
      evt["data"]["key_events"] = json::array();
      evt["data"]["appearance_analyses"] = json::array();
      {
        json aa;
        aa["analysis_id"] = completed_analyses[0].value("analysis_id", "");
        aa["appearance_id"] = completed_analyses[0].value("appearance_id", "");
        aa["summary"] = summary;
        aa["video_url"] = completed_analyses[0].value("video_url", "");
        evt["data"]["appearance_analyses"].push_back(aa);
      }
      evt["data"]["confidence"] = 0.80;
      pushEvent(evt);
    } else {
      // Multiple appearances - send to LLM for aggregation
      std::string prompt = buildDailyAggregationPrompt(
          person_id, person_name, date_key, completed_analyses);

      // Update to inferring
      db->updateBehaviorAnalysisStatus(job_id, "inferring", 85);

      json prog2;
      prog2["event"] = "behavior.analysis.progress";
      prog2["data"]["job_id"] = job_id;
      prog2["data"]["person_id"] = person_id;
      prog2["data"]["analysis_type"] = "daily_aggregation";
      prog2["data"]["status"] = "inferring";
      prog2["data"]["progress_pct"] = 85;
      pushEvent(prog2);

      // Send to cloud
      json cloud_msg;
      cloud_msg["event"] = "behavior.analysis.daily_aggregation";
      cloud_msg["job_id"] = job_id;
      cloud_msg["person_id"] = person_id;
      cloud_msg["person_name"] = person_name;
      cloud_msg["date_key"] = date_key;
      cloud_msg["prompt"] = prompt;
      cloud_msg["child_analyses"] = child_ids;

      std::cout << "[BehaviorAnalysis] Daily aggregation sent to cloud: "
                << "person=" << person_id << " job_id=" << job_id
                << " appearance_count=" << completed_count << std::endl;

      MediaAnalysisServer::GetInstance()->send_to_cloud_client(
          cloud_msg.dump());
    }
  }
}

// ==================== Event Handling ====================

json BehaviorAnalysisTask::handle_event(const json& request,
                                        const std::string& description) {
  if (!request.contains("payload") || !request["payload"].contains("event")) {
    json response;
    response["type"] = "error";
    response["source"] = "c_backend";
    response["destination"] = "web_client";
    response["payload"]["event"] = "error";
    response["payload"]["result"] = "error";
    response["payload"]["error"] = {{"code", "INVALID_REQUEST"},
                                    {"message", "Missing payload.event"}};
    return response;
  }

  std::string event = request["payload"]["event"].get<std::string>();

  if (event == "behavior.analysis.generate") {
    return handleGenerateRequest(request);
  }
  if (event == "behavior.analysis.daily_aggregation") {
    return handleDailyAggregationRequest(request);
  }
  if (event == "video.analysis.generate") {
    return handleVideoAnalysisRequest(request);
  }
  // Handle cloud client responses
  if (event == "behavior.analysis" || event == "report.generate") {
    return handleCloudResponse(request);
  }

  // Default: acknowledge
  json response;
  response["type"] = "event";
  response["source"] = "c_backend";
  response["destination"] = "web_client";
  response["payload"]["event"] = "behavior_analysis";
  response["payload"]["status"] = "received";
  return response;
}

json BehaviorAnalysisTask::handleGenerateRequest(const json& request) {
  json response;
  response["schema_version"] = "smart_home.ws.v1";
  response["type"] = "ack";
  response["source"] = "c_backend";
  response["destination"] = "web_client";
  response["payload"]["event"] = "behavior.analysis.generate";

  if (request.contains("request_id")) {
    response["request_id"] = request["request_id"];
  }

  auto params = request["payload"].value("params", json::object());
  std::string appearance_id = params.value("appearance_id", "");

  if (appearance_id.empty()) {
    response["type"] = "error";
    response["payload"]["result"] = "error";
    response["payload"]["error"] = {{"code", "INVALID_REQUEST"},
                                    {"message", "Missing appearance_id"}};
    return response;
  }

  auto* db = SmartHomeDB::GetInstance();
  json app = db->getAppearance(appearance_id);
  if (app.empty()) {
    response["type"] = "error";
    response["payload"]["result"] = "error";
    response["payload"]["error"] = {{"code", "APPEARANCE_NOT_FOUND"},
                                    {"message", "Appearance not found"}};
    return response;
  }

  std::string job_id = generateJobId("job_per_app_retry");

  response["payload"]["result"] = "accepted";
  response["payload"]["data"]["job_id"] = job_id;

  return response;
}

json BehaviorAnalysisTask::handleDailyAggregationRequest(const json& request) {
  json response;
  response["schema_version"] = "smart_home.ws.v1";
  response["type"] = "ack";
  response["source"] = "c_backend";
  response["destination"] = "web_client";
  response["payload"]["event"] = "behavior.analysis.daily_aggregation";

  if (request.contains("request_id")) {
    response["request_id"] = request["request_id"];
  }

  auto params = request["payload"].value("params", json::object());
  std::string date_key = params.value("date_key", "");

  triggerDailyAggregation(date_key);

  response["payload"]["result"] = "accepted";
  response["payload"]["data"]["status"] = "triggered";
  response["payload"]["data"]["date_key"] =
      date_key.empty() ? getDateKey() : date_key;
  return response;
}

json BehaviorAnalysisTask::handleVideoAnalysisRequest(const json& request) {
  json response;
  response["schema_version"] = "smart_home.ws.v1";
  response["type"] = "ack";
  response["source"] = "c_backend";
  response["destination"] = "web_client";
  response["payload"]["event"] = "video.analysis.generate";

  if (request.contains("request_id")) {
    response["request_id"] = request["request_id"];
  }

  auto params = request["payload"].value("params", json::object());
  std::string channel_id = params.value("channel_id", "0");

  std::string job_id = generateJobId("job_video");

  response["payload"]["result"] = "accepted";
  response["payload"]["data"]["job_id"] = job_id;
  response["payload"]["data"]["channel_id"] = channel_id;

  return response;
}

json BehaviorAnalysisTask::handleCloudResponse(const json& request) {
  // Handle responses from the cloud LLM client
  // These come back as events with analysis results
  std::string event = request["payload"]["event"].get<std::string>();

  if (event == "behavior.analysis") {
    // Cloud returned a per_appearance analysis result
    std::string job_id = request.value("job_id", "");
    std::string summary_text = request.value("analysis_text", "");
    float confidence = request.value("confidence", 0.0f);
    std::string key_frame_json = request.value("key_frame_json", "[]");

    if (!job_id.empty() && !summary_text.empty()) {
      completeAnalysis(job_id, summary_text, key_frame_json, confidence);
    }
  } else if (event == "report.generate") {
    // Cloud returned a report - forward to frontend
    json fwd;
    fwd["event"] = "report.ready";
    fwd["data"] = request.value("payload", json::object());
    pushEvent(fwd);
  }

  return json();  // No direct response needed
}