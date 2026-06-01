#pragma once

#include <json.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::json;

class SmartHomeDB {
 public:
  static SmartHomeDB* GetInstance();

  bool init(const std::string& db_path);
  void close();

  // ---- persons ----
  bool insertPerson(const json& person);
  bool updatePerson(const json& person);
  bool upsertPerson(const json& person);
  json getPerson(const std::string& person_id);
  json listPersons(const std::string& identity_state = "all",
                   const std::string& sort_by = "last_seen_ms",
                   const std::string& order = "desc", int page = 1,
                   int page_size = 20);
  int getPersonCount();
  json searchPersonNames();
  int getMaxRegisteredId();

  // ---- appearances ----
  bool insertAppearance(const json& appearance);
  bool updateAppearanceEnd(const std::string& appearance_id,
                           int64_t end_time_ms, int64_t duration_ms,
                           int last_frame_id, const std::string& stats_json);
  json getAppearance(const std::string& appearance_id);
  json listAppearances(const std::string& person_id, int page = 1,
                       int page_size = 50);

  // ---- snapshots ----
  bool insertSnapshot(const json& snapshot);
  json listSnapshots(const std::string& person_id,
                     const std::string& date_key = "", int page = 1,
                     int page_size = 50);
  json searchSnapshotsByFace(const std::string& person_id = "",
                             const std::string& display_name = "",
                             int registered_id = -1, int page = 1,
                             int page_size = 20);
  std::string getLatestSnapshotImage(const std::string& person_id);

  // ---- alerts ----
  bool insertAlert(const json& alert);
  bool ackAlert(const std::string& alert_id);
  json listAlerts(const std::string& alert_type = "all",
                  const std::string& status = "all", int page = 1,
                  int page_size = 20);
  json getRecentAlerts(int limit = 5);

  // ---- behavior_analyses ----
  bool insertBehaviorAnalysis(const json& analysis);
  bool updateBehaviorAnalysisStatus(const std::string& job_id,
                                    const std::string& status,
                                    int progress_pct = -1,
                                    const std::string& error_code = "",
                                    const std::string& error_message = "");
  bool completeBehaviorAnalysis(const std::string& job_id,
                                const std::string& summary_text,
                                const std::string& key_frame_json,
                                float confidence, int64_t expire_at_ms = 0);
  json getBehaviorAnalysis(const std::string& job_id);
  json listBehaviorAnalyses(const std::string& person_id,
                            const std::string& date_key = "",
                            const std::string& analysis_type = "");
  json getPerAppearanceAnalyses(const std::string& person_id,
                                const std::string& date_key);

  // ---- reports ----
  bool insertReport(const json& report);
  json listReports(const std::string& report_type = "all", int page = 1,
                   int page_size = 20);

  // ---- person stats (Feat-1) ----
  json getPersonStats(const std::string& person_id);
  json getPersonTimeline(const std::string& person_id,
                         const std::string& date_key, int page = 1,
                         int page_size = 50);
  json getDashboardStats();

  // ---- utility ----
  std::string generatePersonId();

 private:
  SmartHomeDB() = default;
  ~SmartHomeDB() { close(); }
  SmartHomeDB(const SmartHomeDB&) = delete;
  SmartHomeDB& operator=(const SmartHomeDB&) = delete;

  bool createTables();
  bool execSQL(const std::string& sql);
  static std::string jsonToString(const json& j);
  static std::string escapeString(const std::string& s);

  void* db_ = nullptr;  // sqlite3*
  std::mutex mutex_;
  bool initialized_ = false;
};