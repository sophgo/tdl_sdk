// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
/**
 * @file anti_facespoofing.hpp
 * @ingroup NETWORKS_FD
 */
#pragma once
#include <chrono>
#include <vector>

#include "core/qnnctx.hpp"
#include "opencv2/opencv.hpp"

namespace qnn {
namespace vision {

/**
 * @brief A spoofing filter container
 *
 * @tparam T Spoofing filter class
 */
template <class T>
struct SpoofingFilter {
    /**
     * @brief Default constructor for a new SpoofingFilter object
     *
     */
    SpoofingFilter() {}

    /**
     * @brief NN based filter constructor for a new SpoofingFilter object
     *
     * @param model_path NN model path
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager
     */
    SpoofingFilter(const std::string &model_path, QNNCtx *qnn_ctx = nullptr)
        : inst(model_path, qnn_ctx) {}
    bool do_filter = true; /**< Filter do flag */
    float weighting = 0.0;
    T inst;                /**< The filter instnace */
};

enum SPOOFINGMODULES {
    CV_BACKGROUND = 0,
    CV_SHARPNESS,
    CV_LBPSVM,
    NN_CLASSIFY,
    NN_CLASSIFY_HSV_YCBCR,
    NN_DEPTH,
    NN_OPTFLOW,
    NN_PATCH,
    NN_CANNY,
    SPOOFINGTOTALMODULES
};

/**
 * @brief Filter value pairs used by AntiFaceSpoofing
 *
 */
using TrackerPair = std::pair<SPOOFINGMODULES, bool>;
using TrackerFilterVec = std::vector<TrackerPair>;

/**
 * @brief Internal tracker queue structure for TrackerQueue class
 *
 */
struct TrackerQueueElements {
    TrackerQueueElements(const cv::Rect &roi, const TrackerFilterVec &val) : region(roi) {
        dequeue.push_back(val);
    }
    cv::Rect region;                      /**< The region on the given input frame */
    std::deque<TrackerFilterVec> dequeue; /**< Filter queue */
    size_t life = 0;                      /**< How long the element hasn't been updated */
};

class TrackerQueue {
    public:
    /**
     * @brief Get the bool flag result of real face
     *
     * @param roi The face region
     * @param filter_values The filter result pairs
     * @param is_real The output bool flag
     * @param use_simple Don't use queue to vote result if set to true
     */
    void GetTrakerResult(const cv::Rect &roi, const short &working_filter_num,
                         const TrackerFilterVec filter_values, bool &is_real,
                         bool use_simple = false);
    private:
    float m_iou_threshold =
        0.7; /**< Input roi will be pushed into one of the m_trackers if roi exceeds this value */
    ushort m_outdated_threshold =
        5; /**< Any m_trackers that doesn't update more that this value will be erased */
    ushort m_queue_decision_length = 10; /**< Any queue inside a m_trackers element will be passed
                                            if its length is smaller than this value */
    std::deque<TrackerQueueElements> m_trackers; /**< The queue contains roi information and its
                                                    corresponding TrackerFilterVec queue */
};
}  // namespace vision
}  // namespace qnn