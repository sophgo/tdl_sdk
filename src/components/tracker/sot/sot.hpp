#include "kalman_box_tracker.hpp"
#include "model/base_model.hpp"
#include "tdl_model_factory.hpp"
#include "tracker/tracker_types.hpp"

struct SOTInfo {
  std::vector<float> bbox;
  std::vector<float> kalman_bbox;
  std::vector<float> template_bbox;
  float score;
  float score_ratio;
  float iou;
  float size_ratio;
  float confidence_of_occluded;
  float confidence_of_reappear;
  int frame_id;
  bool is_occluded;
  bool is_reappear;
};

class SOT : public Tracker {
 public:
  SOT();
  ~SOT();

  int32_t setModel(std::shared_ptr<BaseModel> sot_model) override;

  int32_t initialize(const std::shared_ptr<BaseImage>& image,
                     const std::vector<ObjectBoxInfo>& detect_boxes,
                     const ObjectBoxInfo& bbox, int frame_type) override;

  int32_t initialize(const std::shared_ptr<BaseImage>& image,
                     const std::vector<ObjectBoxInfo>& detect_boxes, float x,
                     float y, int frame_type) override;
  int32_t initialize(const std::shared_ptr<BaseImage>& image,
                     const std::vector<ObjectBoxInfo>& detect_boxes,
                     int index) override;
  int32_t track(const std::shared_ptr<BaseImage>& image, uint64_t frame_id,
                TrackerInfo& tracker_info);

 private:
  // 预处理图像，提取模板和搜索区域
  int32_t initBBox(const std::shared_ptr<BaseImage>& image,
                   const ObjectBoxInfo& init_bbox);

  std::shared_ptr<BaseImage> preprocess(const std::shared_ptr<BaseImage>& image,
                                        const std::vector<float>& bbox,
                                        float offset, int crop_size,
                                        std::vector<int>& context);

  void updateScoreLst(float score);

  // 计算跟踪结果置信度
  void getStatus(const std::vector<float>& bbox,
                 const std::vector<float>& kalman_bbox, float score,
                 float score_ratio, float iou, float size_ratio);

  void ensureBBoxBoundaries(std::vector<float>& bbox,
                            const std::shared_ptr<BaseImage>& image);

  void clampBBox(std::vector<float>& bbox,
                 const std::shared_ptr<BaseImage>& image, int min_side = 3);

  // 模型
  std::shared_ptr<BaseModel> sot_model_;

  // 卡尔曼滤波器
  std::shared_ptr<KalmanBoxTracker> kalman_tracker_;

  // 预处理器
  std::shared_ptr<BasePreprocessor> preprocessor_;

  // 模型参数
  int instance_size_ = 256;           // 实例大小
  int template_size_ = 128;           // 模板大小
  float template_bbox_offset_ = 0.2;  // 模板边界框偏移
  float search_bbox_offset_ = 2.0;    // 搜索边界框偏移
  int kalman_update_count_ = 25;      // 卡尔曼开始更新的帧数
  float size_ratio_threshold_ = 1;    // 宽高比阈值
  float max_expand_ratio_ = 2.0;      // 目标丢失时最大外扩比例

  // 判断目标是否丢失相关参数
  float occluded_score_ratio_threshold_ = 0.9;  // 目标丢失时得分比率阈值
  float occluded_score_threshold_ = 0.6;        // 目标丢失时得分阈值
  float occluded_iou_threshold_ = 0.9;          // 目标丢失时IoU阈值
  float occluded_threshold_ = 0.1;              // 遮挡阈值

  // 判断目标是否重现相关参数
  float reappear_score_threshold_ = 0.3;      // 目标重现时得分阈值
  int reappear_score_ratio_threshold_ = 0.3;  // 目标重现时得分比率阈值
  float reappear_iou_threshold_ = 0.3;        // 目标重现时IoU阈值
  float reappear_threshold_ = 2;              // 重现阈值

  // 当前跟踪的目标边界框 [x, y, w, h]
  std::vector<float> current_bbox_;

  // 模板图像
  std::shared_ptr<BaseImage> template_image_;

  // 是否已初始化
  bool is_initialized_ = false;
  int frame_id_ = 0;
  std::deque<float> score_lst_;
  float score_ratio_ = 1.0f;
  int last_template_update_frame_ = 0;
  int template_update_count_ = 0;
  float prev_w_h_ratio_ = 0.0f;
  TrackStatus status_ = TrackStatus::TRACKED;
  std::vector<float> last_reliable_template_bbox_;
  int lost_frames_ = 0;

  // 中间结果
  SOTInfo sot_info_;
};