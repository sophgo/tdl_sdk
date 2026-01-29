#pragma once
#include <memory>
#include <string>
#include <vector>
#include "components/media_analysis/media_analysis_task.hpp"
#include "matcher/base_matcher.hpp"
#include "tdl_model_factory.hpp"
#include "utils/tokenizer_bpe.hpp"

class ImageTextTask : public MediaAnalysisTask {
 public:
  ImageTextTask(const std::string& data_path, const std::string& model_dir,
                const std::string& txt_dir);
  virtual ~ImageTextTask() = default;

  std::string get_event_type() const override {
    return "image_and_text_matching";
  }
  json handle_event(const json& request,
                    const std::string& description) override;

 private:
  void init_text_model(const std::string& model_dir,
                       const std::string& txt_dir);
  void get_gallery_features(
      std::vector<std::shared_ptr<ModelFeatureInfo>>& gallery_features,
      std::vector<std::string>& image_paths);
  void clip_score_post_process(std::vector<std::vector<float>>& scores);
  void parse_person_info(const std::string& image_path, nlohmann::json& item);

  std::string data_path_;
  std::shared_ptr<BaseMatcher> matcher_;
  std::shared_ptr<BaseModel> text_model_;
  std::shared_ptr<BytePairEncoder> bpe_;
};
