#include "reports_task.hpp"

#include <chrono>
#include <iostream>
#include <sstream>
#include "components/media_analysis/media_analysis_server.hpp"
#include "smart_home_db.hpp"

ReportsTask::ReportsTask() {}

json ReportsTask::buildResponse(const json& request, const std::string& event) {
  json response;
  response["schema_version"] = "smart_home.ws.v1";
  response["message_id"] =
      "srv_" + std::to_string(
                   std::chrono::system_clock::now().time_since_epoch().count());
  if (request.contains("request_id")) {
    response["request_id"] = request["request_id"];
  }
  response["timestamp_ms"] =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  response["type"] = "response";
  response["source"] = "c_backend";
  response["destination"] = "web_client";
  response["payload"]["event"] = event;
  return response;
}

json ReportsTask::handle_event(const json& request,
                               const std::string& description) {
  if (!request.contains("payload") || !request["payload"].contains("event")) {
    json err = buildResponse(request, "error");
    err["type"] = "error";
    err["payload"]["result"] = "error";
    err["payload"]["error"] = {{"code", "INVALID_REQUEST"},
                               {"message", "Missing payload.event"}};
    return err;
  }

  std::string event = request["payload"]["event"].get<std::string>();

  if (event == "reports.list") return handleReportsList(request);
  if (event == "report.generate") return handleReportGenerate(request);

  json err = buildResponse(request, event);
  err["type"] = "error";
  err["payload"]["result"] = "error";
  err["payload"]["error"] = {{"code", "UNKNOWN_EVENT"},
                             {"message", "Unknown event: " + event}};
  return err;
}

json ReportsTask::handleReportsList(const json& request) {
  json response = buildResponse(request, "reports.list");

  auto params = request["payload"].value("params", json::object());
  std::string report_type = params.value("report_type", "all");
  int page = params.value("page", 1);
  int page_size = params.value("page_size", 20);

  auto* db = SmartHomeDB::GetInstance();
  json data = db->listReports(report_type, page, page_size);

  response["payload"]["result"] = "ok";
  response["payload"]["data"] = data;
  return response;
}

json ReportsTask::handleReportGenerate(const json& request) {
  json response = buildResponse(request, "report.generate");

  auto params = request["payload"].value("params", json::object());
  std::string report_type = params.value("report_type", "daily");
  std::string date_key = params.value("date_key", "");

  if (date_key.empty()) {
    // Default to today
    time_t t = time(nullptr);
    struct tm tm_buf;
    localtime_r(&t, &tm_buf);
    char buf[16];
    strftime(buf, sizeof(buf), "%Y-%m-%d", &tm_buf);
    date_key = buf;
  }

  json result;
  std::string report_id;

  if (report_type == "daily") {
    result = generateDailyReport(date_key);
  } else if (report_type == "weekly") {
    result = generateWeeklyReport(date_key);
  } else {
    response["payload"]["result"] = "error";
    response["payload"]["error"] = {
        {"code", "INVALID_REQUEST"},
        {"message", "Unsupported report_type: " + report_type}};
    return response;
  }

  response["type"] = "ack";
  response["payload"]["result"] = "accepted";
  response["payload"]["data"] = result;
  return response;
}

std::string ReportsTask::buildDailyReportPrompt(const std::string& date_key,
                                                const json& all_stats) {
  std::ostringstream prompt;
  prompt << "你是一个智能家居行为分析助手。以下是 " << date_key
         << " 的Smart Home全天活动数据：\n\n";

  prompt << "## 人员出现统计\n";
  if (all_stats.contains("persons") && all_stats["persons"].is_array()) {
    for (const auto& p : all_stats["persons"]) {
      prompt << "- " << p.value("display_name", "未知") << "：出现 "
             << p.value("appearance_count", 0) << " 次\n";
    }
  }

  prompt << "\n## 行为分析摘要\n";
  if (all_stats.contains("analyses") && all_stats["analyses"].is_array()) {
    for (const auto& a : all_stats["analyses"]) {
      prompt << "- " << a.value("summary_text", "") << "\n";
    }
  }

  prompt << "\n## 告警记录\n";
  if (all_stats.contains("alerts") && all_stats["alerts"].is_array()) {
    for (const auto& a : all_stats["alerts"]) {
      prompt << "- [" << a.value("alert_type", "") << "] "
             << a.value("title", "") << ": " << a.value("description", "")
             << "\n";
    }
  }

  prompt << "\n请基于以上数据，生成一份Smart Home日报，包括：\n"
         << "1. 今日活动概览（总共出现几次，主要在哪些区域活动）\n"
         << "2. 今日人员出现统计\n"
         << "3. 异常事件汇总\n"
         << "4. 今日行为摘要\n"
         << "\n请用简洁的中文输出，控制在200字以内。";

  return prompt.str();
}

json ReportsTask::generateDailyReport(const std::string& date_key) {
  auto* db = SmartHomeDB::GetInstance();

  // Gather all data for the report
  json all_stats;

  // Get all persons with appearance counts
  json persons_data = db->listPersons("all", "last_seen_ms", "desc", 1, 50);
  all_stats["persons"] = persons_data["items"];

  // Get all completed behavior analyses for this date
  json analyses;
  analyses["items"] = json::array();
  for (auto& person : persons_data["items"]) {
    std::string pid = person["person_id"].get<std::string>();
    json p_analyses = db->getPerAppearanceAnalyses(pid, date_key);
    for (auto& a : p_analyses["items"]) {
      if (a.value("status", "") == "completed") {
        analyses["items"].push_back(a);
      }
    }
  }
  all_stats["analyses"] = analyses["items"];

  // Get recent alerts
  json alerts_data = db->listAlerts("all", "open", 1, 20);
  all_stats["alerts"] = alerts_data["items"];

  // Get dashboard stats
  json dashboard = db->getDashboardStats();
  all_stats["dashboard"] = dashboard;

  // Build the prompt
  std::string prompt = buildDailyReportPrompt(date_key, all_stats);

  // Generate report_id and create the report entry
  time_t t = time(nullptr);
  int64_t now_ms = (int64_t)t * 1000;

  char report_id_buf[64];
  snprintf(report_id_buf, sizeof(report_id_buf), "r_daily_%s_%lld",
           date_key.c_str(), (long long)t);
  std::string report_id = report_id_buf;

  // Insert into database
  json report;
  report["report_id"] = report_id;
  report["report_type"] = "daily";
  report["date_key"] = date_key;
  report["title"] = "Smart Home 日报 - " + date_key;
  report["summary_text"] = "生成中...";
  report["content_markdown"] = prompt;
  report["attachment_json"] = all_stats.dump();
  report["created_at_ms"] = now_ms;

  db->insertReport(report);

  // Send the prompt to cloud LLM for report generation
  json cloud_msg;
  cloud_msg["event"] = "report.generate";
  cloud_msg["report_id"] = report_id;
  cloud_msg["report_type"] = "daily";
  cloud_msg["date_key"] = date_key;
  cloud_msg["prompt"] = prompt;

  std::cout << "[ReportsTask] Generating daily report for " << date_key
            << std::endl;
  MediaAnalysisServer::GetInstance()->send_to_cloud_client(cloud_msg.dump());

  json result;
  result["report_id"] = report_id;
  result["report_type"] = "daily";
  result["date_key"] = date_key;
  result["status"] = "generating";
  return result;
}

json ReportsTask::generateWeeklyReport(const std::string& week_start_date) {
  auto* db = SmartHomeDB::GetInstance();

  time_t t = time(nullptr);
  int64_t now_ms = (int64_t)t * 1000;

  char report_id_buf[64];
  snprintf(report_id_buf, sizeof(report_id_buf), "r_weekly_%s_%lld",
           week_start_date.c_str(), (long long)t);
  std::string report_id = report_id_buf;

  // Insert placeholder report
  json report;
  report["report_id"] = report_id;
  report["report_type"] = "weekly";
  report["date_key"] = week_start_date;
  report["title"] = "Smart Home 周报 - " + week_start_date;
  report["summary_text"] = "生成中...";
  report["content_markdown"] = "";
  report["attachment_json"] = "[]";
  report["created_at_ms"] = now_ms;

  db->insertReport(report);

  // Build prompt
  std::ostringstream prompt;
  prompt << "你是一个智能家居行为分析助手。以下是 " << week_start_date
         << " 这一周的Smart Home活动汇总数据。\n\n";

  // Get all daily reports for this week
  json daily_reports = db->listReports("daily", 1, 7);
  for (auto& r : daily_reports["items"]) {
    prompt << "## " << r.value("title", "") << "\n";
    prompt << r.value("content_markdown", "") << "\n\n";
  }

  prompt << "\n请基于以上各日报，生成一份Smart Home周报，包括：\n"
         << "1. 本周活动概览\n"
         << "2. 各人员本周行为趋势\n"
         << "3. 异常事件汇总与趋势\n"
         << "4. 值得关注的事项\n"
         << "\n请用简洁的中文输出，控制在300字以内。";

  // Send to cloud
  json cloud_msg;
  cloud_msg["event"] = "report.generate";
  cloud_msg["report_id"] = report_id;
  cloud_msg["report_type"] = "weekly";
  cloud_msg["date_key"] = week_start_date;
  cloud_msg["prompt"] = prompt.str();

  std::cout << "[ReportsTask] Generating weekly report for " << week_start_date
            << std::endl;
  MediaAnalysisServer::GetInstance()->send_to_cloud_client(cloud_msg.dump());

  json result;
  result["report_id"] = report_id;
  result["report_type"] = "weekly";
  result["date_key"] = week_start_date;
  result["status"] = "generating";
  return result;
}

void ReportsTask::sendReportToCloud(const json& report) {
  json msg;
  msg["event"] = "report.generate";
  msg["report"] = report;
  MediaAnalysisServer::GetInstance()->send_to_cloud_client(msg.dump());
}