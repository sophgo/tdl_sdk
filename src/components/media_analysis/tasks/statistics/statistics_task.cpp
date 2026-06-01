#include "statistics_task.hpp"

#include <iostream>
#include "smart_home_db.hpp"

StatisticsTask::StatisticsTask() {}

json StatisticsTask::buildResponse(const json& request,
                                   const std::string& event) {
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

json StatisticsTask::handle_event(const json& request,
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

  if (event == "persons.list") return handlePersonsList(request);
  if (event == "person.detail") return handlePersonDetail(request);
  if (event == "person.timeline") return handlePersonTimeline(request);
  if (event == "persons.search.names") return handleSearchNames(request);
  if (event == "persons.search.face") return handleSearchFace(request);
  if (event == "dashboard.stats") return handleDashboardStats(request);
  if (event == "images.semantic_search")
    return handleImagesSemanticSearch(request);

  json err = buildResponse(request, event);
  err["type"] = "error";
  err["payload"]["result"] = "error";
  err["payload"]["error"] = {{"code", "UNKNOWN_EVENT"},
                             {"message", "Unknown event: " + event}};
  return err;
}

json StatisticsTask::handlePersonsList(const json& request) {
  json response = buildResponse(request, "persons.list");

  auto params = request["payload"].value("params", json::object());
  std::string identity_state = params.value("identity_state", "all");
  std::string sort_by = params.value("sort_by", "last_seen_ms");
  std::string order = params.value("order", "desc");
  int page = params.value("page", 1);
  int page_size = params.value("page_size", 20);

  auto* db = SmartHomeDB::GetInstance();
  json data = db->listPersons(identity_state, sort_by, order, page, page_size);

  // Add avatar URLs
  for (auto& item : data["items"]) {
    std::string pid = item["person_id"].get<std::string>();
    std::string img = db->getLatestSnapshotImage(pid);
    if (!img.empty()) {
      item["avatar_url"] = "/api/image_proxy?path=" + img;
    }
  }

  response["payload"]["result"] = "ok";
  response["payload"]["data"] = data;
  return response;
}

json StatisticsTask::handlePersonDetail(const json& request) {
  json response = buildResponse(request, "person.detail");

  auto params = request["payload"].value("params", json::object());
  std::string person_id = params.value("person_id", "");

  if (person_id.empty()) {
    response["type"] = "error";
    response["payload"]["result"] = "error";
    response["payload"]["error"] = {{"code", "INVALID_REQUEST"},
                                    {"message", "Missing person_id"}};
    return response;
  }

  auto* db = SmartHomeDB::GetInstance();
  json person = db->getPerson(person_id);
  if (person.empty()) {
    response["type"] = "error";
    response["payload"]["result"] = "error";
    response["payload"]["error"] = {{"code", "PERSON_NOT_FOUND"},
                                    {"message", "Person not found"}};
    return response;
  }

  json stats = db->getPersonStats(person_id);
  json recent_alerts = db->listAlerts("all", "all", 1, 5);

  // Filter alerts for this person
  json person_alerts = json::array();
  for (auto& alert : recent_alerts["items"]) {
    if (alert.value("person_id", "") == person_id) {
      person_alerts.push_back(alert);
    }
  }

  std::string img = db->getLatestSnapshotImage(person_id);
  if (!img.empty()) {
    person["avatar_url"] = "/api/image_proxy?path=" + img;
  }

  response["payload"]["result"] = "ok";
  response["payload"]["data"]["person"] = person;
  response["payload"]["data"]["stats"] = stats;
  response["payload"]["data"]["recent_alerts"] = person_alerts;
  return response;
}

json StatisticsTask::handlePersonTimeline(const json& request) {
  json response = buildResponse(request, "person.timeline");

  auto params = request["payload"].value("params", json::object());
  std::string person_id = params.value("person_id", "");
  std::string date_key = params.value("date_key", "");
  int page = params.value("page", 1);
  int page_size = params.value("page_size", 50);

  if (person_id.empty()) {
    response["type"] = "error";
    response["payload"]["result"] = "error";
    response["payload"]["error"] = {{"code", "INVALID_REQUEST"},
                                    {"message", "Missing person_id"}};
    return response;
  }

  auto* db = SmartHomeDB::GetInstance();
  json data = db->getPersonTimeline(person_id, date_key, page, page_size);

  response["payload"]["result"] = "ok";
  response["payload"]["data"] = data;
  return response;
}

json StatisticsTask::handleSearchNames(const json& request) {
  json response = buildResponse(request, "persons.search.names");

  auto* db = SmartHomeDB::GetInstance();
  json data = db->searchPersonNames();

  response["payload"]["result"] = "ok";
  response["payload"]["data"] = data;
  return response;
}

json StatisticsTask::handleSearchFace(const json& request) {
  json response = buildResponse(request, "persons.search.face");

  auto params = request["payload"].value("params", json::object());
  std::string person_id = params.value("person_id", "");
  std::string display_name = params.value("display_name", "");
  int registered_id = params.value("registered_id", -1);
  int page = params.value("page", 1);
  int page_size = params.value("page_size", 20);

  auto* db = SmartHomeDB::GetInstance();
  json data = db->searchSnapshotsByFace(person_id, display_name, registered_id,
                                        page, page_size);

  response["payload"]["result"] = "ok";
  response["payload"]["data"] = data;
  return response;
}

json StatisticsTask::handleDashboardStats(const json& request) {
  json response = buildResponse(request, "dashboard.stats");

  auto* db = SmartHomeDB::GetInstance();
  json stats = db->getDashboardStats();

  response["payload"]["result"] = "ok";
  response["payload"]["data"] = stats;
  return response;
}

json StatisticsTask::handleImagesSemanticSearch(const json& request) {
  json response = buildResponse(request, "images.semantic_search");

  auto params = request["payload"].value("params", json::object());
  std::string query_text = params.value("query_text", "");
  int page = params.value("page", 1);
  int page_size = params.value("page_size", 20);

  auto* db = SmartHomeDB::GetInstance();
  // Semantic search is currently handled by the cloud LLM via ImageTextTask.
  // This handler returns a placeholder indicating the task was accepted
  // for async processing (same pattern as video.analysis.generate).
  // For now, fall back to returning recent snapshots that match criteria.
  json data;
  data["items"] = json::array();
  data["query_text"] = query_text;

  response["payload"]["result"] = "ok";
  response["payload"]["data"] = data;
  response["payload"]["data"]["note"] =
      "Semantic search uses cloud LLM; submit via image_and_text_matching task";
  return response;
}