#include "smart_home_db.hpp"

#include <sqlite3.h>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <sstream>

SmartHomeDB* SmartHomeDB::GetInstance() {
  static SmartHomeDB instance;
  return &instance;
}

std::string SmartHomeDB::escapeString(const std::string& s) {
  char* escaped = sqlite3_mprintf("%q", s.c_str());
  std::string result(escaped);
  sqlite3_free(escaped);
  return result;
}

bool SmartHomeDB::execSQL(const std::string& sql) {
  sqlite3* d = static_cast<sqlite3*>(db_);
  char* err = nullptr;
  int rc = sqlite3_exec(d, sql.c_str(), nullptr, nullptr, &err);
  if (rc != SQLITE_OK) {
    std::cerr << "[SmartHomeDB] SQL error: " << (err ? err : "unknown")
              << "\nSQL: " << sql.substr(0, 200) << std::endl;
    if (err) sqlite3_free(err);
    return false;
  }
  return true;
}

bool SmartHomeDB::createTables() {
  const char* schema[] = {
      // persons
      "CREATE TABLE IF NOT EXISTS persons ("
      "  person_id TEXT PRIMARY KEY,"
      "  registered_id INTEGER UNIQUE,"
      "  display_name TEXT NOT NULL DEFAULT '',"
      "  identity_state TEXT NOT NULL DEFAULT 'unknown',"
      "  source_type TEXT NOT NULL DEFAULT 'auto',"
      "  first_seen_ms INTEGER,"
      "  last_seen_ms INTEGER,"
      "  registration_time_ms INTEGER,"
      "  avatar_snapshot_id TEXT,"
      "  note TEXT NOT NULL DEFAULT '',"
      "  created_at_ms INTEGER NOT NULL,"
      "  updated_at_ms INTEGER NOT NULL"
      ")",

      "CREATE INDEX IF NOT EXISTS idx_persons_state_last_seen "
      "  ON persons(identity_state, last_seen_ms DESC)",

      // appearances
      "CREATE TABLE IF NOT EXISTS appearances ("
      "  appearance_id TEXT PRIMARY KEY,"
      "  person_id TEXT NOT NULL,"
      "  channel_id TEXT NOT NULL,"
      "  track_id INTEGER NOT NULL,"
      "  start_time_ms INTEGER NOT NULL,"
      "  end_time_ms INTEGER,"
      "  duration_ms INTEGER,"
      "  first_frame_id INTEGER,"
      "  last_frame_id INTEGER,"
      "  zone_summary_json TEXT NOT NULL DEFAULT '{}',"
      "  stats_json TEXT NOT NULL DEFAULT '{}',"
      "  created_at_ms INTEGER NOT NULL,"
      "  updated_at_ms INTEGER NOT NULL,"
      "  FOREIGN KEY (person_id) REFERENCES persons(person_id)"
      ")",

      "CREATE INDEX IF NOT EXISTS idx_appearances_person_time "
      "  ON appearances(person_id, start_time_ms DESC)",

      "CREATE INDEX IF NOT EXISTS idx_appearances_channel_track "
      "  ON appearances(channel_id, track_id, start_time_ms DESC)",

      // snapshots
      "CREATE TABLE IF NOT EXISTS snapshots ("
      "  snapshot_id TEXT PRIMARY KEY,"
      "  person_id TEXT NOT NULL,"
      "  appearance_id TEXT,"
      "  channel_id TEXT NOT NULL,"
      "  frame_id INTEGER,"
      "  track_id INTEGER,"
      "  pair_track_id INTEGER,"
      "  object_type TEXT NOT NULL DEFAULT 'face',"
      "  capture_time_ms INTEGER NOT NULL,"
      "  registered_id_at_capture INTEGER,"
      "  identity_state_at_capture TEXT NOT NULL DEFAULT 'unknown',"
      "  quality REAL,"
      "  emotion TEXT,"
      "  attrs_json TEXT NOT NULL DEFAULT '{}',"
      "  image_path TEXT NOT NULL,"
      "  thumbnail_path TEXT NOT NULL DEFAULT '',"
      "  feature_path TEXT NOT NULL DEFAULT '',"
      "  bbox_json TEXT NOT NULL,"
      "  created_at_ms INTEGER NOT NULL,"
      "  FOREIGN KEY (person_id) REFERENCES persons(person_id),"
      "  FOREIGN KEY (appearance_id) REFERENCES appearances(appearance_id)"
      ")",

      "CREATE INDEX IF NOT EXISTS idx_snapshots_person_time "
      "  ON snapshots(person_id, capture_time_ms DESC)",

      "CREATE INDEX IF NOT EXISTS idx_snapshots_appearance "
      "  ON snapshots(appearance_id, capture_time_ms ASC)",

      // alerts
      "CREATE TABLE IF NOT EXISTS alerts ("
      "  alert_id TEXT PRIMARY KEY,"
      "  alert_type TEXT NOT NULL,"
      "  severity TEXT NOT NULL DEFAULT 'info',"
      "  person_id TEXT,"
      "  appearance_id TEXT,"
      "  snapshot_id TEXT,"
      "  channel_id TEXT NOT NULL,"
      "  event_time_ms INTEGER NOT NULL,"
      "  title TEXT NOT NULL,"
      "  description TEXT NOT NULL DEFAULT '',"
      "  detail_json TEXT NOT NULL DEFAULT '{}',"
      "  status TEXT NOT NULL DEFAULT 'open',"
      "  acknowledged_by TEXT NOT NULL DEFAULT '',"
      "  acknowledged_at_ms INTEGER,"
      "  created_at_ms INTEGER NOT NULL,"
      "  updated_at_ms INTEGER NOT NULL,"
      "  FOREIGN KEY (person_id) REFERENCES persons(person_id),"
      "  FOREIGN KEY (appearance_id) REFERENCES appearances(appearance_id),"
      "  FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id)"
      ")",

      "CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts(event_time_ms "
      "DESC)",
      "CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status, "
      "event_time_ms DESC)",
      "CREATE INDEX IF NOT EXISTS idx_alerts_person ON alerts(person_id, "
      "event_time_ms DESC)",

      // behavior_analyses
      "CREATE TABLE IF NOT EXISTS behavior_analyses ("
      "  analysis_id TEXT PRIMARY KEY,"
      "  job_id TEXT NOT NULL DEFAULT '',"
      "  person_id TEXT NOT NULL,"
      "  appearance_id TEXT,"
      "  date_key TEXT NOT NULL,"
      "  analysis_type TEXT NOT NULL DEFAULT 'per_appearance',"
      "  trigger_type TEXT NOT NULL DEFAULT 'auto',"
      "  status TEXT NOT NULL DEFAULT 'accepted',"
      "  progress_pct INTEGER NOT NULL DEFAULT 0,"
      "  error_code TEXT NOT NULL DEFAULT '',"
      "  error_message TEXT NOT NULL DEFAULT '',"
      "  key_frame_json TEXT NOT NULL DEFAULT '[]',"
      "  summary_text TEXT NOT NULL DEFAULT '',"
      "  video_url TEXT NOT NULL DEFAULT '',"
      "  model_name TEXT NOT NULL DEFAULT '',"
      "  prompt_version TEXT NOT NULL DEFAULT '',"
      "  confidence REAL,"
      "  expire_at_ms INTEGER,"
      "  created_at_ms INTEGER NOT NULL,"
      "  FOREIGN KEY (person_id) REFERENCES persons(person_id),"
      "  FOREIGN KEY (appearance_id) REFERENCES appearances(appearance_id)"
      ")",

      "CREATE INDEX IF NOT EXISTS idx_behavior_person_date "
      "  ON behavior_analyses(person_id, date_key DESC, created_at_ms DESC)",

      "CREATE INDEX IF NOT EXISTS idx_behavior_job_id "
      "  ON behavior_analyses(job_id)",

      "CREATE INDEX IF NOT EXISTS idx_behavior_appearance "
      "  ON behavior_analyses(appearance_id)",

      // reports
      "CREATE TABLE IF NOT EXISTS reports ("
      "  report_id TEXT PRIMARY KEY,"
      "  report_type TEXT NOT NULL,"
      "  date_key TEXT NOT NULL,"
      "  title TEXT NOT NULL,"
      "  summary_text TEXT NOT NULL,"
      "  content_markdown TEXT NOT NULL,"
      "  attachment_json TEXT NOT NULL DEFAULT '[]',"
      "  created_at_ms INTEGER NOT NULL"
      ")",

      "CREATE INDEX IF NOT EXISTS idx_reports_type_date "
      "  ON reports(report_type, date_key DESC)",

      // cooccurrences
      "CREATE TABLE IF NOT EXISTS cooccurrences ("
      "  cooccurrence_id TEXT PRIMARY KEY,"
      "  person_a_id TEXT NOT NULL,"
      "  person_b_id TEXT NOT NULL,"
      "  channel_id TEXT NOT NULL,"
      "  start_time_ms INTEGER NOT NULL,"
      "  end_time_ms INTEGER,"
      "  duration_ms INTEGER,"
      "  zone_name TEXT NOT NULL DEFAULT '',"
      "  created_at_ms INTEGER NOT NULL,"
      "  FOREIGN KEY (person_a_id) REFERENCES persons(person_id),"
      "  FOREIGN KEY (person_b_id) REFERENCES persons(person_id)"
      ")",

      "CREATE INDEX IF NOT EXISTS idx_cooccurrence_pair_time "
      "  ON cooccurrences(person_a_id, person_b_id, start_time_ms DESC)",

      // roi_configs
      "CREATE TABLE IF NOT EXISTS roi_configs ("
      "  roi_id TEXT PRIMARY KEY,"
      "  channel_id TEXT NOT NULL,"
      "  name TEXT NOT NULL,"
      "  roi_type TEXT NOT NULL DEFAULT 'rectangle',"
      "  points_json TEXT NOT NULL,"
      "  rule_json TEXT NOT NULL DEFAULT '{}',"
      "  enabled INTEGER NOT NULL DEFAULT 1,"
      "  created_at_ms INTEGER NOT NULL,"
      "  updated_at_ms INTEGER NOT NULL"
      ")",

      "CREATE INDEX IF NOT EXISTS idx_roi_channel_enabled "
      "  ON roi_configs(channel_id, enabled)",

      // behavior_baselines
      "CREATE TABLE IF NOT EXISTS behavior_baselines ("
      "  person_id TEXT PRIMARY KEY,"
      "  baseline_json TEXT NOT NULL DEFAULT '{}',"
      "  sample_days INTEGER NOT NULL DEFAULT 0,"
      "  last_updated_ms INTEGER NOT NULL,"
      "  FOREIGN KEY (person_id) REFERENCES persons(person_id)"
      ")",
  };

  for (const char* sql : schema) {
    if (!execSQL(sql)) return false;
  }
  return true;
}

bool SmartHomeDB::init(const std::string& db_path) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_) return true;

  sqlite3* d = nullptr;
  int rc = sqlite3_open_v2(
      db_path.c_str(), &d,
      SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
      nullptr);
  if (rc != SQLITE_OK) {
    std::cerr << "[SmartHomeDB] Cannot open database: " << db_path << std::endl;
    return false;
  }
  db_ = d;

  // Enable WAL mode for better concurrent read performance
  execSQL("PRAGMA journal_mode=WAL");
  execSQL("PRAGMA foreign_keys=ON");

  if (!createTables()) {
    std::cerr << "[SmartHomeDB] Failed to create tables" << std::endl;
    sqlite3_close(d);
    db_ = nullptr;
    return false;
  }

  initialized_ = true;
  std::cout << "[SmartHomeDB] Initialized: " << db_path << std::endl;
  return true;
}

void SmartHomeDB::close() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (db_) {
    sqlite3_close(static_cast<sqlite3*>(db_));
    db_ = nullptr;
  }
  initialized_ = false;
}

// ==================== Persons ====================

bool SmartHomeDB::insertPerson(const json& person) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  const char* sql =
      "INSERT OR IGNORE INTO persons (person_id, registered_id, display_name, "
      "identity_state, source_type, first_seen_ms, last_seen_ms, "
      "registration_time_ms, avatar_snapshot_id, note, created_at_ms, "
      "updated_at_ms) "
      "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)";

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

  std::string pid = person.value("person_id", "");
  int64_t now = (int64_t)time(nullptr) * 1000;

  sqlite3_bind_text(stmt, 1, pid.c_str(), -1, SQLITE_TRANSIENT);
  if (person.contains("registered_id") && !person["registered_id"].is_null())
    sqlite3_bind_int(stmt, 2, person["registered_id"].get<int>());
  else
    sqlite3_bind_null(stmt, 2);
  sqlite3_bind_text(stmt, 3, person.value("display_name", "").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 4, person.value("identity_state", "unknown").c_str(),
                    -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 5, person.value("source_type", "auto").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt, 6, person.value("first_seen_ms", now));
  sqlite3_bind_int64(stmt, 7, person.value("last_seen_ms", now));
  if (person.contains("registration_time_ms") &&
      !person["registration_time_ms"].is_null())
    sqlite3_bind_int64(stmt, 8, person["registration_time_ms"].get<int64_t>());
  else
    sqlite3_bind_null(stmt, 8);
  if (person.contains("avatar_snapshot_id") &&
      !person["avatar_snapshot_id"].is_null())
    sqlite3_bind_text(stmt, 9,
                      person["avatar_snapshot_id"].get<std::string>().c_str(),
                      -1, SQLITE_TRANSIENT);
  else
    sqlite3_bind_null(stmt, 9);
  sqlite3_bind_text(stmt, 10, person.value("note", "").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt, 11, person.value("created_at_ms", now));
  sqlite3_bind_int64(stmt, 12, person.value("updated_at_ms", now));

  int rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return rc == SQLITE_DONE;
}

bool SmartHomeDB::updatePerson(const json& person) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  const char* sql =
      "UPDATE persons SET registered_id=?, display_name=?, identity_state=?, "
      "last_seen_ms=?, registration_time_ms=?, avatar_snapshot_id=?, note=?, "
      "updated_at_ms=? WHERE person_id=?";

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

  int64_t now = (int64_t)time(nullptr) * 1000;

  if (person.contains("registered_id") && !person["registered_id"].is_null())
    sqlite3_bind_int(stmt, 1, person["registered_id"].get<int>());
  else
    sqlite3_bind_null(stmt, 1);
  sqlite3_bind_text(stmt, 2, person.value("display_name", "").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 3, person.value("identity_state", "unknown").c_str(),
                    -1, SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt, 4, person.value("last_seen_ms", now));
  if (person.contains("registration_time_ms") &&
      !person["registration_time_ms"].is_null())
    sqlite3_bind_int64(stmt, 5, person["registration_time_ms"].get<int64_t>());
  else
    sqlite3_bind_null(stmt, 5);
  if (person.contains("avatar_snapshot_id") &&
      !person["avatar_snapshot_id"].is_null())
    sqlite3_bind_text(stmt, 6,
                      person["avatar_snapshot_id"].get<std::string>().c_str(),
                      -1, SQLITE_TRANSIENT);
  else
    sqlite3_bind_null(stmt, 6);
  sqlite3_bind_text(stmt, 7, person.value("note", "").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt, 8, now);
  sqlite3_bind_text(stmt, 9, person["person_id"].get<std::string>().c_str(), -1,
                    SQLITE_TRANSIENT);

  int rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return rc == SQLITE_DONE;
}

bool SmartHomeDB::upsertPerson(const json& person) {
  // Try insert first
  json existing = getPerson(person.value("person_id", ""));
  if (existing.empty()) {
    return insertPerson(person);
  }
  return updatePerson(person);
}

json SmartHomeDB::getPerson(const std::string& person_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  const char* sql = "SELECT * FROM persons WHERE person_id=?";
  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql, -1, &stmt, nullptr) != SQLITE_OK)
    return json();

  sqlite3_bind_text(stmt, 1, person_id.c_str(), -1, SQLITE_TRANSIENT);

  json result;
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    result["person_id"] =
        std::string((const char*)sqlite3_column_text(stmt, 0));
    if (sqlite3_column_type(stmt, 1) != SQLITE_NULL)
      result["registered_id"] = sqlite3_column_int(stmt, 1);
    result["display_name"] =
        std::string((const char*)sqlite3_column_text(stmt, 2));
    result["identity_state"] =
        std::string((const char*)sqlite3_column_text(stmt, 3));
    result["source_type"] =
        std::string((const char*)sqlite3_column_text(stmt, 4));
    result["first_seen_ms"] = sqlite3_column_int64(stmt, 5);
    result["last_seen_ms"] = sqlite3_column_int64(stmt, 6);
    if (sqlite3_column_type(stmt, 7) != SQLITE_NULL)
      result["registration_time_ms"] = sqlite3_column_int64(stmt, 7);
    if (sqlite3_column_type(stmt, 8) != SQLITE_NULL)
      result["avatar_snapshot_id"] =
          std::string((const char*)sqlite3_column_text(stmt, 8));
  }
  sqlite3_finalize(stmt);
  return result;
}

json SmartHomeDB::listPersons(const std::string& identity_state,
                              const std::string& sort_by,
                              const std::string& order, int page,
                              int page_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  std::ostringstream sql;
  sql << "SELECT p.person_id, p.registered_id, p.display_name, "
         "p.identity_state, p.source_type, p.first_seen_ms, p.last_seen_ms, "
         "p.registration_time_ms, p.avatar_snapshot_id, "
         "COUNT(a.appearance_id) as appearance_count "
         "FROM persons p "
         "LEFT JOIN appearances a ON p.person_id = a.person_id ";

  if (identity_state != "all") {
    sql << "WHERE p.identity_state='" << escapeString(identity_state) << "' ";
  }

  sql << "GROUP BY p.person_id ";

  // Validate sort_by to prevent injection
  if (sort_by == "last_seen_ms" || sort_by == "first_seen_ms" ||
      sort_by == "display_name" || sort_by == "appearance_count") {
    sql << "ORDER BY " << sort_by << " ";
  } else {
    sql << "ORDER BY p.last_seen_ms ";
  }

  if (order == "asc") {
    sql << "ASC ";
  } else {
    sql << "DESC ";
  }

  sql << "LIMIT " << page_size << " OFFSET " << ((page - 1) * page_size);

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql.str().c_str(), -1, &stmt, nullptr) != SQLITE_OK)
    return json();

  json result;
  result["items"] = json::array();
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    json item;
    item["person_id"] = std::string((const char*)sqlite3_column_text(stmt, 0));
    if (sqlite3_column_type(stmt, 1) != SQLITE_NULL)
      item["registered_id"] = sqlite3_column_int(stmt, 1);
    item["display_name"] =
        std::string((const char*)sqlite3_column_text(stmt, 2));
    item["identity_state"] =
        std::string((const char*)sqlite3_column_text(stmt, 3));
    item["source_type"] =
        std::string((const char*)sqlite3_column_text(stmt, 4));
    item["first_seen_ms"] = sqlite3_column_int64(stmt, 5);
    item["last_seen_ms"] = sqlite3_column_int64(stmt, 6);
    if (sqlite3_column_type(stmt, 7) != SQLITE_NULL)
      item["registration_time_ms"] = sqlite3_column_int64(stmt, 7);
    if (sqlite3_column_type(stmt, 8) != SQLITE_NULL)
      item["avatar_snapshot_id"] =
          std::string((const char*)sqlite3_column_text(stmt, 8));
    item["appearance_count"] = sqlite3_column_int(stmt, 9);
    item["online"] = false;  // set by caller
    result["items"].push_back(item);
  }
  sqlite3_finalize(stmt);

  // Get total count
  std::ostringstream cnt_sql;
  cnt_sql << "SELECT COUNT(DISTINCT p.person_id) FROM persons p ";
  if (identity_state != "all") {
    cnt_sql << "WHERE p.identity_state='" << escapeString(identity_state)
            << "' ";
  }
  sqlite3_stmt* cnt_stmt = nullptr;
  if (sqlite3_prepare_v2(d, cnt_sql.str().c_str(), -1, &cnt_stmt, nullptr) ==
      SQLITE_OK) {
    if (sqlite3_step(cnt_stmt) == SQLITE_ROW) {
      int total = sqlite3_column_int(cnt_stmt, 0);
      result["page"] = {{"page", page},
                        {"page_size", page_size},
                        {"total", total},
                        {"has_more", (page * page_size) < total}};
    }
    sqlite3_finalize(cnt_stmt);
  }

  return result;
}

int SmartHomeDB::getPersonCount() {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);
  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, "SELECT COUNT(*) FROM persons", -1, &stmt,
                         nullptr) != SQLITE_OK)
    return 0;
  int count = 0;
  if (sqlite3_step(stmt) == SQLITE_ROW) count = sqlite3_column_int(stmt, 0);
  sqlite3_finalize(stmt);
  return count;
}

json SmartHomeDB::searchPersonNames() {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  const char* sql =
      "SELECT person_id, registered_id, display_name FROM persons "
      "WHERE identity_state='registered' ORDER BY display_name";
  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql, -1, &stmt, nullptr) != SQLITE_OK)
    return json();

  json result;
  result["items"] = json::array();
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    json item;
    item["person_id"] = std::string((const char*)sqlite3_column_text(stmt, 0));
    if (sqlite3_column_type(stmt, 1) != SQLITE_NULL)
      item["registered_id"] = sqlite3_column_int(stmt, 1);
    item["display_name"] =
        std::string((const char*)sqlite3_column_text(stmt, 2));
    result["items"].push_back(item);
  }
  sqlite3_finalize(stmt);
  return result;
}

int SmartHomeDB::getMaxRegisteredId() {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);
  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d,
                         "SELECT MAX(registered_id) FROM persons WHERE "
                         "registered_id IS NOT NULL",
                         -1, &stmt, nullptr) != SQLITE_OK)
    return 0;
  int max_id = 0;
  if (sqlite3_step(stmt) == SQLITE_ROW &&
      sqlite3_column_type(stmt, 0) != SQLITE_NULL)
    max_id = sqlite3_column_int(stmt, 0);
  sqlite3_finalize(stmt);
  return max_id;
}

// ==================== Appearances ====================

bool SmartHomeDB::insertAppearance(const json& appearance) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  const char* sql =
      "INSERT INTO appearances (appearance_id, person_id, channel_id, "
      "track_id, start_time_ms, end_time_ms, duration_ms, first_frame_id, "
      "last_frame_id, zone_summary_json, stats_json, created_at_ms, "
      "updated_at_ms) "
      "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)";

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

  int64_t now = (int64_t)time(nullptr) * 1000;

  sqlite3_bind_text(stmt, 1,
                    appearance["appearance_id"].get<std::string>().c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 2, appearance["person_id"].get<std::string>().c_str(),
                    -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 3,
                    appearance["channel_id"].get<std::string>().c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_int(stmt, 4, appearance["track_id"].get<int>());
  sqlite3_bind_int64(stmt, 5, appearance["start_time_ms"].get<int64_t>());
  sqlite3_bind_null(stmt, 6);  // end_time_ms
  sqlite3_bind_null(stmt, 7);  // duration_ms
  sqlite3_bind_int(stmt, 8, appearance.value("first_frame_id", 0));
  sqlite3_bind_null(stmt, 9);  // last_frame_id
  sqlite3_bind_text(stmt, 10, "{}", -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 11, "{}", -1, SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt, 12, appearance.value("created_at_ms", now));
  sqlite3_bind_int64(stmt, 13, appearance.value("updated_at_ms", now));

  int rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return rc == SQLITE_DONE;
}

bool SmartHomeDB::updateAppearanceEnd(const std::string& appearance_id,
                                      int64_t end_time_ms, int64_t duration_ms,
                                      int last_frame_id,
                                      const std::string& stats_json) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  const char* sql =
      "UPDATE appearances SET end_time_ms=?, duration_ms=?, last_frame_id=?, "
      "stats_json=?, updated_at_ms=? WHERE appearance_id=?";

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

  int64_t now = (int64_t)time(nullptr) * 1000;

  sqlite3_bind_int64(stmt, 1, end_time_ms);
  sqlite3_bind_int64(stmt, 2, duration_ms);
  sqlite3_bind_int(stmt, 3, last_frame_id);
  sqlite3_bind_text(stmt, 4, stats_json.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt, 5, now);
  sqlite3_bind_text(stmt, 6, appearance_id.c_str(), -1, SQLITE_TRANSIENT);

  int rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return rc == SQLITE_DONE;
}

json SmartHomeDB::getAppearance(const std::string& appearance_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  const char* sql = "SELECT * FROM appearances WHERE appearance_id=?";
  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql, -1, &stmt, nullptr) != SQLITE_OK)
    return json();

  sqlite3_bind_text(stmt, 1, appearance_id.c_str(), -1, SQLITE_TRANSIENT);

  json result;
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    result["appearance_id"] =
        std::string((const char*)sqlite3_column_text(stmt, 0));
    result["person_id"] =
        std::string((const char*)sqlite3_column_text(stmt, 1));
    result["channel_id"] =
        std::string((const char*)sqlite3_column_text(stmt, 2));
    result["track_id"] = sqlite3_column_int(stmt, 3);
    result["start_time_ms"] = sqlite3_column_int64(stmt, 4);
    if (sqlite3_column_type(stmt, 5) != SQLITE_NULL)
      result["end_time_ms"] = sqlite3_column_int64(stmt, 5);
    if (sqlite3_column_type(stmt, 6) != SQLITE_NULL)
      result["duration_ms"] = sqlite3_column_int64(stmt, 6);
  }
  sqlite3_finalize(stmt);
  return result;
}

json SmartHomeDB::listAppearances(const std::string& person_id, int page,
                                  int page_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  const char* sql =
      "SELECT appearance_id, person_id, channel_id, track_id, start_time_ms, "
      "end_time_ms, duration_ms, stats_json "
      "FROM appearances WHERE person_id=? ORDER BY start_time_ms DESC "
      "LIMIT ? OFFSET ?";

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql, -1, &stmt, nullptr) != SQLITE_OK)
    return json();

  sqlite3_bind_text(stmt, 1, person_id.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_int(stmt, 2, page_size);
  sqlite3_bind_int(stmt, 3, (page - 1) * page_size);

  json result;
  result["items"] = json::array();
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    json item;
    item["appearance_id"] =
        std::string((const char*)sqlite3_column_text(stmt, 0));
    item["person_id"] = std::string((const char*)sqlite3_column_text(stmt, 1));
    item["channel_id"] = std::string((const char*)sqlite3_column_text(stmt, 2));
    item["track_id"] = sqlite3_column_int(stmt, 3);
    item["start_time_ms"] = sqlite3_column_int64(stmt, 4);
    if (sqlite3_column_type(stmt, 5) != SQLITE_NULL)
      item["end_time_ms"] = sqlite3_column_int64(stmt, 5);
    if (sqlite3_column_type(stmt, 6) != SQLITE_NULL)
      item["duration_ms"] = sqlite3_column_int64(stmt, 6);
    if (sqlite3_column_type(stmt, 7) != SQLITE_NULL)
      item["stats_json"] =
          std::string((const char*)sqlite3_column_text(stmt, 7));
    result["items"].push_back(item);
  }
  sqlite3_finalize(stmt);
  return result;
}

// ==================== Snapshots ====================

bool SmartHomeDB::insertSnapshot(const json& snapshot) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  const char* sql =
      "INSERT INTO snapshots (snapshot_id, person_id, appearance_id, "
      "channel_id, frame_id, track_id, pair_track_id, object_type, "
      "capture_time_ms, registered_id_at_capture, identity_state_at_capture, "
      "quality, emotion, attrs_json, image_path, thumbnail_path, "
      "feature_path, bbox_json, created_at_ms) "
      "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)";

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

  int64_t now = (int64_t)time(nullptr) * 1000;

  sqlite3_bind_text(stmt, 1, snapshot["snapshot_id"].get<std::string>().c_str(),
                    -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 2, snapshot["person_id"].get<std::string>().c_str(),
                    -1, SQLITE_TRANSIENT);
  if (snapshot.contains("appearance_id") &&
      !snapshot["appearance_id"].is_null())
    sqlite3_bind_text(stmt, 3,
                      snapshot["appearance_id"].get<std::string>().c_str(), -1,
                      SQLITE_TRANSIENT);
  else
    sqlite3_bind_null(stmt, 3);
  sqlite3_bind_text(stmt, 4, snapshot["channel_id"].get<std::string>().c_str(),
                    -1, SQLITE_TRANSIENT);
  sqlite3_bind_int(stmt, 5, snapshot.value("frame_id", 0));
  sqlite3_bind_int(stmt, 6, snapshot.value("track_id", 0));
  if (snapshot.contains("pair_track_id") &&
      !snapshot["pair_track_id"].is_null())
    sqlite3_bind_int(stmt, 7, snapshot["pair_track_id"].get<int>());
  else
    sqlite3_bind_null(stmt, 7);
  sqlite3_bind_text(stmt, 8, snapshot.value("object_type", "face").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt, 9, snapshot["capture_time_ms"].get<int64_t>());
  if (snapshot.contains("registered_id_at_capture") &&
      !snapshot["registered_id_at_capture"].is_null())
    sqlite3_bind_int(stmt, 10, snapshot["registered_id_at_capture"].get<int>());
  else
    sqlite3_bind_null(stmt, 10);
  sqlite3_bind_text(
      stmt, 11, snapshot.value("identity_state_at_capture", "unknown").c_str(),
      -1, SQLITE_TRANSIENT);
  if (snapshot.contains("quality") && !snapshot["quality"].is_null())
    sqlite3_bind_double(stmt, 12, snapshot["quality"].get<double>());
  else
    sqlite3_bind_null(stmt, 12);
  if (snapshot.contains("emotion") && !snapshot["emotion"].is_null())
    sqlite3_bind_text(stmt, 13, snapshot["emotion"].get<std::string>().c_str(),
                      -1, SQLITE_TRANSIENT);
  else
    sqlite3_bind_null(stmt, 13);
  sqlite3_bind_text(stmt, 14, snapshot.value("attrs_json", "{}").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 15, snapshot["image_path"].get<std::string>().c_str(),
                    -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 16, snapshot.value("thumbnail_path", "").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 17, snapshot.value("feature_path", "").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 18, snapshot.value("bbox_json", "{}").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt, 19, snapshot.value("created_at_ms", now));

  int rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return rc == SQLITE_DONE;
}

json SmartHomeDB::listSnapshots(const std::string& person_id,
                                const std::string& date_key, int page,
                                int page_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  std::ostringstream sql;
  sql << "SELECT snapshot_id, person_id, appearance_id, channel_id, frame_id, "
         "track_id, object_type, capture_time_ms, quality, emotion, "
         "image_path, thumbnail_path, bbox_json "
         "FROM snapshots WHERE person_id='"
      << escapeString(person_id) << "' ";

  if (!date_key.empty()) {
    // date_key is like "2026-05-09"
    // capture_time_ms falls within that day
    sql << "AND date(capture_time_ms/1000, 'unixepoch')='"
        << escapeString(date_key) << "' ";
  }

  sql << "ORDER BY capture_time_ms DESC "
      << "LIMIT " << page_size << " OFFSET " << ((page - 1) * page_size);

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql.str().c_str(), -1, &stmt, nullptr) != SQLITE_OK)
    return json();

  json result;
  result["items"] = json::array();
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    json item;
    item["snapshot_id"] =
        std::string((const char*)sqlite3_column_text(stmt, 0));
    item["person_id"] = std::string((const char*)sqlite3_column_text(stmt, 1));
    if (sqlite3_column_type(stmt, 2) != SQLITE_NULL)
      item["appearance_id"] =
          std::string((const char*)sqlite3_column_text(stmt, 2));
    item["channel_id"] = std::string((const char*)sqlite3_column_text(stmt, 3));
    item["frame_id"] = sqlite3_column_int(stmt, 4);
    item["track_id"] = sqlite3_column_int(stmt, 5);
    item["object_type"] =
        std::string((const char*)sqlite3_column_text(stmt, 6));
    item["capture_time_ms"] = sqlite3_column_int64(stmt, 7);
    if (sqlite3_column_type(stmt, 8) != SQLITE_NULL)
      item["quality"] = sqlite3_column_double(stmt, 8);
    if (sqlite3_column_type(stmt, 9) != SQLITE_NULL)
      item["emotion"] = std::string((const char*)sqlite3_column_text(stmt, 9));
    item["image_url"] = "/api/image_proxy?path=" +
                        std::string((const char*)sqlite3_column_text(stmt, 10));
    if (sqlite3_column_type(stmt, 11) != SQLITE_NULL &&
        strlen((const char*)sqlite3_column_text(stmt, 11)) > 0)
      item["thumbnail_url"] =
          "/api/image_proxy?path=" +
          std::string((const char*)sqlite3_column_text(stmt, 11));
    if (sqlite3_column_type(stmt, 12) != SQLITE_NULL)
      item["bbox_json"] =
          std::string((const char*)sqlite3_column_text(stmt, 12));
    result["items"].push_back(item);
  }
  sqlite3_finalize(stmt);
  return result;
}

json SmartHomeDB::searchSnapshotsByFace(const std::string& person_id,
                                        const std::string& display_name,
                                        int registered_id, int page,
                                        int page_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  std::ostringstream sql;
  sql << "SELECT s.snapshot_id, s.person_id, s.channel_id, s.frame_id, "
         "s.track_id, s.object_type, s.capture_time_ms, s.emotion, "
         "s.image_path, s.thumbnail_path, p.display_name "
         "FROM snapshots s "
         "LEFT JOIN persons p ON s.person_id = p.person_id WHERE 1=1 ";

  if (!person_id.empty()) {
    sql << "AND s.person_id='" << escapeString(person_id) << "' ";
  }
  if (!display_name.empty()) {
    sql << "AND p.display_name='" << escapeString(display_name) << "' ";
  }
  if (registered_id >= 0) {
    sql << "AND s.registered_id_at_capture=" << registered_id << " ";
  }

  sql << "ORDER BY s.capture_time_ms DESC "
      << "LIMIT " << page_size << " OFFSET " << ((page - 1) * page_size);

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql.str().c_str(), -1, &stmt, nullptr) != SQLITE_OK)
    return json();

  json result;
  result["items"] = json::array();
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    json item;
    item["snapshot_id"] =
        std::string((const char*)sqlite3_column_text(stmt, 0));
    item["person_id"] = std::string((const char*)sqlite3_column_text(stmt, 1));
    item["channel_id"] = std::string((const char*)sqlite3_column_text(stmt, 2));
    item["frame_id"] = sqlite3_column_int(stmt, 3);
    item["track_id"] = sqlite3_column_int(stmt, 4);
    item["object_type"] =
        std::string((const char*)sqlite3_column_text(stmt, 5));
    item["capture_time_ms"] = sqlite3_column_int64(stmt, 6);
    if (sqlite3_column_type(stmt, 7) != SQLITE_NULL)
      item["emotion"] = std::string((const char*)sqlite3_column_text(stmt, 7));
    item["image_url"] = "/api/image_proxy?path=" +
                        std::string((const char*)sqlite3_column_text(stmt, 8));
    if (sqlite3_column_type(stmt, 9) != SQLITE_NULL &&
        strlen((const char*)sqlite3_column_text(stmt, 9)) > 0)
      item["thumbnail_url"] =
          "/api/image_proxy?path=" +
          std::string((const char*)sqlite3_column_text(stmt, 9));
    if (sqlite3_column_type(stmt, 10) != SQLITE_NULL)
      item["display_name"] =
          std::string((const char*)sqlite3_column_text(stmt, 10));
    result["items"].push_back(item);
  }
  sqlite3_finalize(stmt);
  return result;
}

std::string SmartHomeDB::getLatestSnapshotImage(const std::string& person_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  const char* sql =
      "SELECT image_path FROM snapshots WHERE person_id=? "
      "ORDER BY capture_time_ms DESC LIMIT 1";
  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql, -1, &stmt, nullptr) != SQLITE_OK) return "";

  sqlite3_bind_text(stmt, 1, person_id.c_str(), -1, SQLITE_TRANSIENT);

  std::string path;
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    path = std::string((const char*)sqlite3_column_text(stmt, 0));
  }
  sqlite3_finalize(stmt);
  return path;
}

// ==================== Alerts ====================

bool SmartHomeDB::insertAlert(const json& alert) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  const char* sql =
      "INSERT INTO alerts (alert_id, alert_type, severity, person_id, "
      "appearance_id, snapshot_id, channel_id, event_time_ms, title, "
      "description, detail_json, status, acknowledged_by, "
      "acknowledged_at_ms, created_at_ms, updated_at_ms) "
      "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)";

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

  int64_t now = (int64_t)time(nullptr) * 1000;

  sqlite3_bind_text(stmt, 1, alert["alert_id"].get<std::string>().c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 2, alert["alert_type"].get<std::string>().c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 3, alert.value("severity", "info").c_str(), -1,
                    SQLITE_TRANSIENT);
  if (alert.contains("person_id") && !alert["person_id"].is_null())
    sqlite3_bind_text(stmt, 4, alert["person_id"].get<std::string>().c_str(),
                      -1, SQLITE_TRANSIENT);
  else
    sqlite3_bind_null(stmt, 4);
  if (alert.contains("appearance_id") && !alert["appearance_id"].is_null())
    sqlite3_bind_text(stmt, 5,
                      alert["appearance_id"].get<std::string>().c_str(), -1,
                      SQLITE_TRANSIENT);
  else
    sqlite3_bind_null(stmt, 5);
  if (alert.contains("snapshot_id") && !alert["snapshot_id"].is_null())
    sqlite3_bind_text(stmt, 6, alert["snapshot_id"].get<std::string>().c_str(),
                      -1, SQLITE_TRANSIENT);
  else
    sqlite3_bind_null(stmt, 6);
  sqlite3_bind_text(stmt, 7, alert["channel_id"].get<std::string>().c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt, 8, alert["event_time_ms"].get<int64_t>());
  sqlite3_bind_text(stmt, 9, alert["title"].get<std::string>().c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 10, alert.value("description", "").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 11, alert.value("detail_json", "{}").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 12, alert.value("status", "open").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 13, "", -1, SQLITE_TRANSIENT);
  sqlite3_bind_null(stmt, 14);
  sqlite3_bind_int64(stmt, 15, alert.value("created_at_ms", now));
  sqlite3_bind_int64(stmt, 16, alert.value("updated_at_ms", now));

  int rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return rc == SQLITE_DONE;
}

bool SmartHomeDB::ackAlert(const std::string& alert_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  int64_t now = (int64_t)time(nullptr) * 1000;
  char sql[256];
  snprintf(sql, sizeof(sql),
           "UPDATE alerts SET status='acknowledged', acknowledged_at_ms=%lld, "
           "updated_at_ms=%lld WHERE alert_id='%s'",
           (long long)now, (long long)now, escapeString(alert_id).c_str());
  return execSQL(sql);
}

json SmartHomeDB::listAlerts(const std::string& alert_type,
                             const std::string& status, int page,
                             int page_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  std::ostringstream sql;
  sql << "SELECT a.alert_id, a.alert_type, a.severity, a.person_id, "
         "a.appearance_id, a.snapshot_id, a.channel_id, a.event_time_ms, "
         "a.title, a.description, a.detail_json, a.status, "
         "p.display_name "
         "FROM alerts a "
         "LEFT JOIN persons p ON a.person_id = p.person_id WHERE 1=1 ";

  if (alert_type != "all") {
    sql << "AND a.alert_type='" << escapeString(alert_type) << "' ";
  }
  if (status != "all") {
    sql << "AND a.status='" << escapeString(status) << "' ";
  }

  sql << "ORDER BY a.event_time_ms DESC "
      << "LIMIT " << page_size << " OFFSET " << ((page - 1) * page_size);

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql.str().c_str(), -1, &stmt, nullptr) != SQLITE_OK)
    return json();

  json result;
  result["items"] = json::array();
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    json item;
    item["alert_id"] = std::string((const char*)sqlite3_column_text(stmt, 0));
    item["alert_type"] = std::string((const char*)sqlite3_column_text(stmt, 1));
    item["severity"] = std::string((const char*)sqlite3_column_text(stmt, 2));
    if (sqlite3_column_type(stmt, 3) != SQLITE_NULL)
      item["person_id"] =
          std::string((const char*)sqlite3_column_text(stmt, 3));
    item["event_time_ms"] = sqlite3_column_int64(stmt, 7);
    item["title"] = std::string((const char*)sqlite3_column_text(stmt, 8));
    item["description"] =
        std::string((const char*)sqlite3_column_text(stmt, 9));
    item["status"] = std::string((const char*)sqlite3_column_text(stmt, 11));
    if (sqlite3_column_type(stmt, 12) != SQLITE_NULL)
      item["display_name"] =
          std::string((const char*)sqlite3_column_text(stmt, 12));
    result["items"].push_back(item);
  }
  sqlite3_finalize(stmt);
  return result;
}

json SmartHomeDB::getRecentAlerts(int limit) {
  return listAlerts("all", "all", 1, limit);
}

// ==================== Behavior Analyses ====================

bool SmartHomeDB::insertBehaviorAnalysis(const json& analysis) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  const char* sql =
      "INSERT INTO behavior_analyses (analysis_id, job_id, person_id, "
      "appearance_id, date_key, analysis_type, trigger_type, status, "
      "progress_pct, error_code, error_message, key_frame_json, "
      "summary_text, video_url, model_name, prompt_version, "
      "confidence, expire_at_ms, created_at_ms) "
      "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)";

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

  int64_t now = (int64_t)time(nullptr) * 1000;

  sqlite3_bind_text(stmt, 1, analysis["analysis_id"].get<std::string>().c_str(),
                    -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 2, analysis.value("job_id", "").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 3, analysis["person_id"].get<std::string>().c_str(),
                    -1, SQLITE_TRANSIENT);
  if (analysis.contains("appearance_id") &&
      !analysis["appearance_id"].is_null())
    sqlite3_bind_text(stmt, 4,
                      analysis["appearance_id"].get<std::string>().c_str(), -1,
                      SQLITE_TRANSIENT);
  else
    sqlite3_bind_null(stmt, 4);
  sqlite3_bind_text(stmt, 5, analysis["date_key"].get<std::string>().c_str(),
                    -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 6,
                    analysis.value("analysis_type", "per_appearance").c_str(),
                    -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 7, analysis.value("trigger_type", "auto").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 8, analysis.value("status", "accepted").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_int(stmt, 9, analysis.value("progress_pct", 0));
  sqlite3_bind_text(stmt, 10, analysis.value("error_code", "").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 11, analysis.value("error_message", "").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 12, analysis.value("key_frame_json", "[]").c_str(),
                    -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 13, analysis.value("summary_text", "").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 14, analysis.value("video_url", "").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 15, analysis.value("model_name", "").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 16, analysis.value("prompt_version", "").c_str(), -1,
                    SQLITE_TRANSIENT);
  if (analysis.contains("confidence") && !analysis["confidence"].is_null())
    sqlite3_bind_double(stmt, 17, analysis["confidence"].get<double>());
  else
    sqlite3_bind_null(stmt, 17);
  if (analysis.contains("expire_at_ms") && !analysis["expire_at_ms"].is_null())
    sqlite3_bind_int64(stmt, 18, analysis["expire_at_ms"].get<int64_t>());
  else
    sqlite3_bind_null(stmt, 18);
  sqlite3_bind_int64(stmt, 19, analysis.value("created_at_ms", now));

  int rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return rc == SQLITE_DONE;
}

bool SmartHomeDB::updateBehaviorAnalysisStatus(
    const std::string& job_id, const std::string& status, int progress_pct,
    const std::string& error_code, const std::string& error_message) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  std::ostringstream sql;
  sql << "UPDATE behavior_analyses SET status='" << escapeString(status) << "'";
  if (progress_pct >= 0) {
    sql << ", progress_pct=" << progress_pct;
  }
  if (!error_code.empty()) {
    sql << ", error_code='" << escapeString(error_code) << "'";
  }
  if (!error_message.empty()) {
    sql << ", error_message='" << escapeString(error_message) << "'";
  }
  sql << " WHERE job_id='" << escapeString(job_id) << "'";

  return execSQL(sql.str());
}

bool SmartHomeDB::completeBehaviorAnalysis(const std::string& job_id,
                                           const std::string& summary_text,
                                           const std::string& key_frame_json,
                                           float confidence,
                                           int64_t expire_at_ms) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  const char* sql =
      "UPDATE behavior_analyses SET status='completed', progress_pct=100, "
      "summary_text=?, key_frame_json=?, confidence=?, expire_at_ms=? "
      "WHERE job_id=?";

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

  sqlite3_bind_text(stmt, 1, summary_text.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 2, key_frame_json.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_double(stmt, 3, confidence);
  if (expire_at_ms > 0)
    sqlite3_bind_int64(stmt, 4, expire_at_ms);
  else
    sqlite3_bind_null(stmt, 4);
  sqlite3_bind_text(stmt, 5, job_id.c_str(), -1, SQLITE_TRANSIENT);

  int rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return rc == SQLITE_DONE;
}

json SmartHomeDB::getBehaviorAnalysis(const std::string& job_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  const char* sql = "SELECT * FROM behavior_analyses WHERE job_id=?";
  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql, -1, &stmt, nullptr) != SQLITE_OK)
    return json();

  sqlite3_bind_text(stmt, 1, job_id.c_str(), -1, SQLITE_TRANSIENT);

  json result;
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    result["analysis_id"] =
        std::string((const char*)sqlite3_column_text(stmt, 0));
    result["job_id"] = std::string((const char*)sqlite3_column_text(stmt, 1));
    result["person_id"] =
        std::string((const char*)sqlite3_column_text(stmt, 2));
    if (sqlite3_column_type(stmt, 3) != SQLITE_NULL)
      result["appearance_id"] =
          std::string((const char*)sqlite3_column_text(stmt, 3));
    result["date_key"] = std::string((const char*)sqlite3_column_text(stmt, 4));
    result["analysis_type"] =
        std::string((const char*)sqlite3_column_text(stmt, 5));
    result["status"] = std::string((const char*)sqlite3_column_text(stmt, 7));
    result["progress_pct"] = sqlite3_column_int(stmt, 8);
    result["summary_text"] =
        std::string((const char*)sqlite3_column_text(stmt, 12));
    result["video_url"] =
        std::string((const char*)sqlite3_column_text(stmt, 13));
    if (sqlite3_column_type(stmt, 16) != SQLITE_NULL)
      result["confidence"] = sqlite3_column_double(stmt, 16);
  }
  sqlite3_finalize(stmt);
  return result;
}

json SmartHomeDB::listBehaviorAnalyses(const std::string& person_id,
                                       const std::string& date_key,
                                       const std::string& analysis_type) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  std::ostringstream sql;
  sql << "SELECT analysis_id, job_id, person_id, appearance_id, date_key, "
         "analysis_type, status, progress_pct, summary_text, video_url, "
         "confidence, key_frame_json, created_at_ms "
         "FROM behavior_analyses WHERE person_id='"
      << escapeString(person_id) << "' ";

  if (!date_key.empty()) {
    sql << "AND date_key='" << escapeString(date_key) << "' ";
  }
  if (!analysis_type.empty()) {
    sql << "AND analysis_type='" << escapeString(analysis_type) << "' ";
  }

  sql << "ORDER BY created_at_ms DESC LIMIT 50";

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql.str().c_str(), -1, &stmt, nullptr) != SQLITE_OK)
    return json();

  json result;
  result["items"] = json::array();
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    json item;
    item["analysis_id"] =
        std::string((const char*)sqlite3_column_text(stmt, 0));
    item["job_id"] = std::string((const char*)sqlite3_column_text(stmt, 1));
    item["person_id"] = std::string((const char*)sqlite3_column_text(stmt, 2));
    if (sqlite3_column_type(stmt, 3) != SQLITE_NULL)
      item["appearance_id"] =
          std::string((const char*)sqlite3_column_text(stmt, 3));
    item["date_key"] = std::string((const char*)sqlite3_column_text(stmt, 4));
    item["analysis_type"] =
        std::string((const char*)sqlite3_column_text(stmt, 5));
    item["status"] = std::string((const char*)sqlite3_column_text(stmt, 6));
    item["progress_pct"] = sqlite3_column_int(stmt, 7);
    item["summary_text"] =
        std::string((const char*)sqlite3_column_text(stmt, 8));
    item["video_url"] = std::string((const char*)sqlite3_column_text(stmt, 9));
    if (sqlite3_column_type(stmt, 10) != SQLITE_NULL)
      item["confidence"] = sqlite3_column_double(stmt, 10);
    item["created_at_ms"] = sqlite3_column_int64(stmt, 12);
    result["items"].push_back(item);
  }
  sqlite3_finalize(stmt);
  return result;
}

json SmartHomeDB::getPerAppearanceAnalyses(const std::string& person_id,
                                           const std::string& date_key) {
  return listBehaviorAnalyses(person_id, date_key, "per_appearance");
}

// ==================== Reports ====================

bool SmartHomeDB::insertReport(const json& report) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  const char* sql =
      "INSERT INTO reports (report_id, report_type, date_key, title, "
      "summary_text, content_markdown, attachment_json, created_at_ms) "
      "VALUES (?,?,?,?,?,?,?,?)";

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

  int64_t now = (int64_t)time(nullptr) * 1000;

  sqlite3_bind_text(stmt, 1, report["report_id"].get<std::string>().c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 2, report["report_type"].get<std::string>().c_str(),
                    -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 3, report["date_key"].get<std::string>().c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 4, report["title"].get<std::string>().c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 5, report.value("summary_text", "").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 6, report.value("content_markdown", "").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 7, report.value("attachment_json", "[]").c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt, 8, report.value("created_at_ms", now));

  int rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return rc == SQLITE_DONE;
}

json SmartHomeDB::listReports(const std::string& report_type, int page,
                              int page_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);

  std::ostringstream sql;
  sql << "SELECT report_id, report_type, date_key, title, summary_text, "
         "content_markdown, attachment_json, created_at_ms "
         "FROM reports ";
  if (report_type != "all") {
    sql << "WHERE report_type='" << escapeString(report_type) << "' ";
  }
  sql << "ORDER BY date_key DESC "
      << "LIMIT " << page_size << " OFFSET " << ((page - 1) * page_size);

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(d, sql.str().c_str(), -1, &stmt, nullptr) != SQLITE_OK)
    return json();

  json result;
  result["items"] = json::array();
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    json item;
    item["report_id"] = std::string((const char*)sqlite3_column_text(stmt, 0));
    item["report_type"] =
        std::string((const char*)sqlite3_column_text(stmt, 1));
    item["date_key"] = std::string((const char*)sqlite3_column_text(stmt, 2));
    item["title"] = std::string((const char*)sqlite3_column_text(stmt, 3));
    item["summary_text"] =
        std::string((const char*)sqlite3_column_text(stmt, 4));
    item["content_markdown"] =
        std::string((const char*)sqlite3_column_text(stmt, 5));
    item["attachment_json"] =
        std::string((const char*)sqlite3_column_text(stmt, 6));
    item["created_at_ms"] = sqlite3_column_int64(stmt, 7);
    result["items"].push_back(item);
  }
  sqlite3_finalize(stmt);
  return result;
}

// ==================== Person Stats (Feat-1) ====================

json SmartHomeDB::getPersonStats(const std::string& person_id) {
  json stats;
  stats["person_id"] = person_id;

  // Get daily appearance counts for last 7 days
  {
    std::lock_guard<std::mutex> lock(mutex_);
    sqlite3* d = static_cast<sqlite3*>(db_);

    const char* sql =
        "SELECT date(start_time_ms/1000, 'unixepoch') as day, COUNT(*) as cnt "
        "FROM appearances WHERE person_id=? "
        "AND start_time_ms > ? "
        "GROUP BY day ORDER BY day ASC";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(d, sql, -1, &stmt, nullptr) == SQLITE_OK) {
      sqlite3_bind_text(stmt, 1, person_id.c_str(), -1, SQLITE_TRANSIENT);
      int64_t seven_days_ago = (int64_t)time(nullptr) * 1000 - 7LL * 86400000LL;
      sqlite3_bind_int64(stmt, 2, seven_days_ago);

      stats["daily_appearances"] = json::array();
      int total = 0;
      while (sqlite3_step(stmt) == SQLITE_ROW) {
        json day;
        day["date"] = std::string((const char*)sqlite3_column_text(stmt, 0));
        int cnt = sqlite3_column_int(stmt, 1);
        day["count"] = cnt;
        total += cnt;
        stats["daily_appearances"].push_back(day);
      }
      stats["total_appearances_7d"] = total;
      sqlite3_finalize(stmt);
    }

    // Get total appearance count
    const char* total_sql =
        "SELECT COUNT(*) FROM appearances WHERE person_id=?";
    if (sqlite3_prepare_v2(d, total_sql, -1, &stmt, nullptr) == SQLITE_OK) {
      sqlite3_bind_text(stmt, 1, person_id.c_str(), -1, SQLITE_TRANSIENT);
      if (sqlite3_step(stmt) == SQLITE_ROW)
        stats["total_appearances"] = sqlite3_column_int(stmt, 0);
      sqlite3_finalize(stmt);
    }

    // Get active zone distribution
    const char* zone_sql =
        "SELECT zone_summary_json FROM appearances WHERE person_id=? "
        "AND zone_summary_json != '{}' LIMIT 50";
    if (sqlite3_prepare_v2(d, zone_sql, -1, &stmt, nullptr) == SQLITE_OK) {
      sqlite3_bind_text(stmt, 1, person_id.c_str(), -1, SQLITE_TRANSIENT);
      stats["active_zones"] = json::array();
      while (sqlite3_step(stmt) == SQLITE_ROW) {
        std::string zone_json((const char*)sqlite3_column_text(stmt, 0));
        try {
          stats["active_zones"].push_back(json::parse(zone_json));
        } catch (...) {
        }
      }
      sqlite3_finalize(stmt);
    };
  }

  return stats;
}

json SmartHomeDB::getPersonTimeline(const std::string& person_id,
                                    const std::string& date_key, int page,
                                    int page_size) {
  json result = listSnapshots(person_id, date_key, page, page_size);
  result["person_id"] = person_id;
  result["date_key"] = date_key;
  return result;
}

json SmartHomeDB::getDashboardStats() {
  json stats;
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3* d = static_cast<sqlite3*>(db_);
  sqlite3_stmt* stmt = nullptr;

  // Total registered persons
  if (sqlite3_prepare_v2(
          d, "SELECT COUNT(*) FROM persons WHERE identity_state='registered'",
          -1, &stmt, nullptr) == SQLITE_OK) {
    if (sqlite3_step(stmt) == SQLITE_ROW)
      stats["registered_count"] = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
  }

  // Total pending persons
  if (sqlite3_prepare_v2(
          d, "SELECT COUNT(*) FROM persons WHERE identity_state='pending'", -1,
          &stmt, nullptr) == SQLITE_OK) {
    if (sqlite3_step(stmt) == SQLITE_ROW)
      stats["pending_count"] = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
  }

  // Total appearances today
  int64_t day_start = (int64_t)(time(nullptr) / 86400) * 86400 * 1000;
  if (sqlite3_prepare_v2(
          d, "SELECT COUNT(*) FROM appearances WHERE start_time_ms >= ?", -1,
          &stmt, nullptr) == SQLITE_OK) {
    sqlite3_bind_int64(stmt, 1, day_start);
    if (sqlite3_step(stmt) == SQLITE_ROW)
      stats["today_appearances"] = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
  }

  // Total alerts (open)
  if (sqlite3_prepare_v2(d, "SELECT COUNT(*) FROM alerts WHERE status='open'",
                         -1, &stmt, nullptr) == SQLITE_OK) {
    if (sqlite3_step(stmt) == SQLITE_ROW)
      stats["open_alerts"] = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
  }

  // 24h activity distribution
  {
    const char* sql =
        "SELECT CAST(strftime('%H', start_time_ms/1000, 'unixepoch') AS "
        "INTEGER) as hour, "
        "COUNT(*) as cnt FROM appearances WHERE start_time_ms >= ? "
        "GROUP BY hour ORDER BY hour";
    if (sqlite3_prepare_v2(d, sql, -1, &stmt, nullptr) == SQLITE_OK) {
      int64_t day_ago = (int64_t)time(nullptr) * 1000 - 86400000LL;
      sqlite3_bind_int64(stmt, 1, day_ago);
      stats["hourly_activity"] = json::array();
      for (int h = 0; h < 24; h++) {
        json hour_entry;
        hour_entry["hour"] = h;
        hour_entry["count"] = 0;
        stats["hourly_activity"].push_back(hour_entry);
      }
      while (sqlite3_step(stmt) == SQLITE_ROW) {
        int hour = sqlite3_column_int(stmt, 0);
        int cnt = sqlite3_column_int(stmt, 1);
        if (hour >= 0 && hour < 24) {
          stats["hourly_activity"][hour]["count"] = cnt;
        }
      }
      sqlite3_finalize(stmt);
    }
  }

  return stats;
}

// ==================== Utility ====================

std::string SmartHomeDB::generatePersonId() {
  time_t t = time(nullptr);
  struct tm tm_buf;
  localtime_r(&t, &tm_buf);
  char date_part[16];
  strftime(date_part, sizeof(date_part), "%Y%m%d", &tm_buf);

  int count = getPersonCount() + 1;
  char id[64];
  snprintf(id, sizeof(id), "p_%s_%06d", date_part, count);
  return std::string(id);
}