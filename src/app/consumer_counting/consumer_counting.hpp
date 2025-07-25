#ifndef CONSUMER_COUNTING_HPP
#define CONSUMER_COUNTING_HPP

#include "components/tracker/tracker_types.hpp"
#include "nn/tdl_model_factory.hpp"

typedef struct {
  float old_x;
  float old_y;
  int unmatched_times;
  int counting_gap;
  bool enter;
  bool miss;

} Consumer;

class ConsumerCounting {
 public:
  ConsumerCounting(int A_x, int A_y, int B_x, int B_y, int mode);

  ~ConsumerCounting() {}

  int32_t update_state(const std::vector<TrackerInfo> &track_results);
  uint32_t get_enter_num() { return enter_num_; };
  uint32_t get_miss_num() { return miss_num_; };

  int32_t consumer_counting(float old_x, float old_y, float cur_x, float cur_y,
                            Consumer &it);

  bool isLineIntersect(float old_x, float old_y, float cur_x, float cur_y);
  float crossProduct(float A_x, float A_y, float B_x, float B_y, float C_x,
                     float C_y);

 private:
  int mode_;
  int A_x_, A_y_, B_x_, B_y_;
  int buffer_width_ = 20;
  int counting_gap_ = 50;
  int MAX_UNMATCHED_TIME = 15;

  uint32_t enter_num_;
  uint32_t miss_num_;

  float normal_vector_x_;
  float normal_vector_y_;

  std::map<uint64_t, Consumer> persons_;
  std::map<uint64_t, Consumer> heads_;
};

#endif
