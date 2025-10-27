#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP
#include <string>
#include <vector>
#include "common/model_output_types.hpp"
#include "tracker/tracker_types.hpp"
class Evaluator {
 public:
  Evaluator();
  ~Evaluator();

  void evaluate(const std::vector<std::string> &eval_files,
                const std::string &output_dir);

  int32_t writeResult(const std::string &result_file,
                      const std::string &str_content);
  std::string packOutput(std::shared_ptr<ModelOutputInfo> model_output);
  std::string packOutput(const std::vector<TrackerInfo> &track_results);

 protected:
  int img_width_;
  int img_height_;
};

#endif
