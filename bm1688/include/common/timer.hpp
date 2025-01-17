#ifndef COMMON_TIMER_HPP_
#define COMMON_TIMER_HPP_

#include <ctime>
#include <map>
#include <string>
#include <vector>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <sys/time.h>

#endif
#ifdef WIN32
int gettimeofday(struct timeval* tp, void* tzp) {
  time_t clock;
  struct tm tm;
  SYSTEMTIME wtm;

  GetLocalTime(&wtm);
  tm.tm_year = wtm.wYear - 1900;
  tm.tm_mon = wtm.wMonth - 1;
  tm.tm_mday = wtm.wDay;
  tm.tm_hour = wtm.wHour;
  tm.tm_min = wtm.wMinute;
  tm.tm_sec = wtm.wSecond;
  tm.tm_isdst = -1;
  clock = mktime(&tm);
  tp->tv_sec = clock;
  tp->tv_usec = wtm.wMilliseconds * 1000;

  return (0);
}
#endif

class TimeRecorder {
 public:
  TimeRecorder() : elapses_(), tags_() { tags_.reserve(2000); }

  ~TimeRecorder() {}

  inline long get_cur_us() {
    struct timeval t0;
    gettimeofday(&t0, NULL);
    return t0.tv_sec * 1000000 + t0.tv_usec;
  }

  void store_timestamp(const std::string& tag) {
    std::map<std::string, long>::iterator it = elapses_.find(tag);
    if (it == elapses_.end()) {
      elapses_[tag] = get_cur_us();
      tags_.push_back(tag);
    } else {
      elapses_[tag] = get_cur_us() - elapses_[tag];
    }
  }

  void store_timestamp(const std::string& tag, int index) {
    std::stringstream ss;
    ss << tag << "#" << index;
    store_timestamp(ss.str());
  }

  void store_timestamp(const std::string& tag, char name, int index) {
    std::stringstream ss;
    ss << tag << "$" << name << "#" << index;
    store_timestamp(ss.str());
  }

  void clear() {
    elapses_.clear();
    tags_.clear();
  }

  void show();

  std::map<std::string, long> elapses_;
  std::vector<std::string> tags_;
};

#endif
