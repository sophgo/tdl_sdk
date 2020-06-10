/**
 * @brief Object detection by Yolo
 *
 */
#pragma once

#include "core/image_net.hpp"
#include "face/face_types.hpp"

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace qnn {
namespace vision {

typedef struct {
    float x, y, w, h;
} box;

typedef struct detection {
    box bbox;
    int classes;
    float *prob;
    float objectness;
    int sort_class;
} detection;

class Yolo2 : public ImageNet {
public:
    /**
     * @brief Construct a new Yolo2 object
     *
     * @param model_path The bmodel path
     * @param dataset_type The dataset used to train the model
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager
     */
    Yolo2(const std::string &model_path, const std::string dataset_type, QNNCtx *qnn_ctx = nullptr);

    /**
     * @brief Set the threshold used by Yolo2
     *
     * @param threshold The confidence threshold for Yolo2. [default: 0.6]
     */
    void SetThreshold(const float threshold);

    /**
     * @brief Detect the objects with input image.
     *
     * @param image The input image
     * @param object detection results
     */
    virtual void Detect(const cv::Mat &image, std::vector<object_detect_rect_t> &object_dets);

    /**
     * @brief A simple helper function to draw the object box on an image.
     *
     * @param object detection results from Yolo2::Detect
     * @param img The input image
     */
    void Draw(const std::vector<object_detect_rect_t> &object_dets, cv::Mat &img);

protected:
    void PrepareImage(const cv::Mat &image, cv::Mat &prepared, float &ratio) override;
    virtual void FillNetworkBoxes(OutputTensor &net_output, int w, int h, float threshold, int relative, detection *dets);
    virtual int GetNumOfDetections(OutputTensor &net_output);
    detection *GetNetworkBoxes(OutputTensor &net_output, int w, int h, float threshold, int relative, int *num);
    void GetRegionDetections(OutputTensor &net_out, int w, int h, int netw, int neth, float threshold, int relative, detection *dets);
    detection *MakeNetworkBoxes(OutputTensor &net_output, float threshold, int *num);
    std::vector<object_detect_rect_t> GetResults(cv::Mat &im, detection *dets, int num, float threshold, std::vector<std::string> &names);
    void DoRegion(OutputTensor &net_output, int batch);

protected:
    int m_classes = 80;
    float m_threshold = 0.6;
    float m_nms_threshold = 0.45;
    int m_num = 5;
    int m_coords = 4;
    int m_batch = 1;
    float* m_biases;
    std::vector<std::string> m_names;
};

class Yolo3 : public Yolo2 {
public:
    /**
     * @brief Construct a new Yolo3 object
     *
     * @param model_path The bmodel path
     * @param dataset_type The dataset used to train the model
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager
     */
    Yolo3(const std::string &model_path, const std::string dataset_type, QNNCtx *qnn_ctx = nullptr);

    /**
     * @brief Detect the objects with input image.
     *
     * @param image The input image
     * @param object detection results
     */
    void Detect(const cv::Mat &image, std::vector<object_detect_rect_t> &object_dets) override;

protected:
    void FillNetworkBoxes(OutputTensor &net_output, int w, int h, float threshold, int relative, detection *dets) override;
    int GetNumOfDetections(OutputTensor &net_output) override;
    void DoYolo(OutputTensor &net_output, int batch);
    void GetYoloDetections(OutputTensor &net_out, int w, int h, int net_width, int net_height, float threshold, int relative, detection *dets);

private:
    // Mask for biases(anchors)
    std::vector<int> m_mask;
};

}  // namespace vision
}  // namespace qnn
