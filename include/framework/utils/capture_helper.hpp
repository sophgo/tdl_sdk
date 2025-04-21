#include "opencv2/core/core.hpp"
#include "model/base_model.hpp"
#include "common/model_output_types.hpp"

// 定义姿态结构体
typedef struct {
    double yaw;    // 偏航角（绕Z轴旋转）
    double pitch;  // 俯仰角（绕Y轴旋转）
    double roll;   // 翻滚角（绕X轴旋转）
    float facialUnitNormalVector[3];
} Pose;

int Predict(std::shared_ptr<ModelLandmarksInfo> pFacial5points,  Pose *hp);

// Camera-centered coordinate system
// x & y axis aligned along the horizontal and vertical directions in the image.
// z axis along the normal to the image plain.
int Predict3DFacialNormal(const cv::Point &noseTip, const cv::Point &noseBase,
                          const cv::Point &midEye, const cv::Point &midMouth,
                          Pose *hp);

float CalDistance(const cv::Point &p1, const cv::Point &p2);
float CalAngle(const cv::Point &pt1, const cv::Point &pt2);
float CalSlant(int ln, int lf, const float Rn, float theta);