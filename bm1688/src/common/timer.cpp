#include "common/timer.hpp"
#include <log/Logger.hpp>

void TimeRecorder::show() {
  for (size_t i = 0; i < tags_.size(); i++) {
    LOG(INFO) << "time<" << i << "> " << tags_[i] << ": "
              << elapses_[tags_[i]] << " us";
  }

}

