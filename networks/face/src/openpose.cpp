// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
#include "openpose.hpp"
#include "openpose/net/nmsBase.hpp"
#include "openpose/pose/poseParameters.hpp"
#include "openpose/utilities/fastMath.hpp"
#include "openpose/utilities/openCv.hpp"

#define OPENPOSE_WIDTH 224
#define OPENPOSE_HEIGHT 128

namespace qnn {
namespace vision {
// clang-format off
// From OpenCV
// connection table, in the format [model_id][pair_id][from/to]
const int POSE_PAIRS[3][20][2] = {
{   // COCO body
    {1,2}, {1,5}, {2,3},
    {3,4}, {5,6}, {6,7},
    {1,8}, {8,9}, {9,10},
    {1,11}, {11,12}, {12,13},
    {1,0}, {0,14},
    {14,16}, {0,15}, {15,17}
},
{   // MPI body
    {0,1}, {1,2}, {2,3},
    {3,4}, {1,5}, {5,6},
    {6,7}, {1,14}, {14,8}, {8,9},
    {9,10}, {14,11}, {11,12}, {12,13}
},
{   // hand
    {0,1}, {1,2}, {2,3}, {3,4},         // thumb
    {0,5}, {5,6}, {6,7}, {7,8},         // pinkie
    {0,9}, {9,10}, {10,11}, {11,12},    // middle
    {0,13}, {13,14}, {14,15}, {15,16},  // ring
    {0,17}, {17,18}, {18,19}, {19,20}   // small
}};
// clang-format on

static const std::vector<NetShape> kPossibleInputShapes = {
    NetShape(1, 3, OPENPOSE_HEIGHT, OPENPOSE_WIDTH)};

Openpose::Openpose(const std::string &model_path, const std::string dataset_type, const bool opt,
                   QNNCtx *qnn_ctx)
    : ImageNet(model_path, kPossibleInputShapes, 3, OPENPOSE_WIDTH, OPENPOSE_HEIGHT, true,
               cv::INTER_LINEAR, qnn_ctx) {
    // Setup dataset_type
    if (!dataset_type.compare("COCO")) {
        m_info.midx = 0;
        m_info.npairs = 17;
        m_info.nparts = 18;
        m_info.pose_model = op::PoseModel::COCO_18;
    } else if (!dataset_type.compare("MPI")) {
        m_info.midx = 1;
        m_info.npairs = 14;
        m_info.nparts = 16;
        m_info.pose_model = op::PoseModel::MPI_15;
    } else if (!dataset_type.compare("HAND")) {
        m_info.midx = 2;
        m_info.npairs = 20;
        m_info.nparts = 22;
        m_info.pose_model = op::PoseModel::BODY_25;
    } else {
        std::cerr << "Can't interpret dataset parameter: " << dataset_type << std::endl;
        exit(-1);
    }
    // Detect multi or single body pose
    m_info.detect_multi = opt;

    // Get threshold and set quantize value
    float threshold = GetInPutThreshold();
    printf("threshold: %f\n", threshold);
    // img = ((img - 128) / 255) * 128 / threshold
    SetQuanParams({std::vector<float>{-128.f, -128.f, -128.f},
                   std::vector<float>{float(128) / (threshold * 255)}});

    // Setup properties
    bool maximizePositives = false;
    m_info.multi_maximize_positives = maximizePositives;
    m_info.properties[(int)op::PoseProperty::NMSThreshold] =
        op::getPoseDefaultNmsThreshold(m_info.pose_model, maximizePositives);
    m_info.properties[(int)op::PoseProperty::ConnectInterMinAboveThreshold] =
        op::getPoseDefaultConnectInterMinAboveThreshold(maximizePositives);
    m_info.properties[(int)op::PoseProperty::ConnectInterThreshold] =
        op::getPoseDefaultConnectInterThreshold(m_info.pose_model, maximizePositives);
    m_info.properties[(int)op::PoseProperty::ConnectMinSubsetCnt] =
        op::getPoseDefaultMinSubsetCnt(maximizePositives);
    m_info.properties[(int)op::PoseProperty::ConnectMinSubsetScore] =
        op::getPoseDefaultConnectMinSubsetScore(maximizePositives);
}

void Openpose::SetThreshold(const float threshold) {
    m_info.properties[(int)op::PoseProperty::ConnectInterThreshold] = threshold;
}

void Openpose::Detect(const cv::Mat &image, std::vector<std::vector<cv::Point2f>> &results) {
    if (image.empty()) {
        return;
    }
    results.clear();
    std::vector<cv::Mat> images;
    images.emplace_back(image);

    ImageNet::Detect(images, [&](OutTensors &out, std::vector<float> &ratios, int start, int end) {
        assert(start >= 0 && start < end && end <= (int)images.size());
        OutputTensor &net_out = out["net_output"];
        float *data = (float *)net_out.data;
        int H = net_out.shape.h;
        int W = net_out.shape.w;
        GetBodies(data, W, H, net_out.shape.c, images[start], results);
    });
}

void Openpose::Draw(const std::vector<std::vector<cv::Point2f>> &input, cv::Mat &img) {
    std::vector<std::vector<BodyStructure>> skeleton;
    GetSkeletonVectors(input, skeleton);
    for (size_t i = 0; i < skeleton.size(); i++) {
        for (size_t j = 0; j < skeleton[i].size(); j++) {
            cv::Point2f a = skeleton[i][j].a;
            cv::Point2f b = skeleton[i][j].b;
            // we did not find enough confidence before
            if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0) continue;

            cv::line(img, a, b, cv::Scalar(0, 200, 0), 2);
            cv::circle(img, a, 3, cv::Scalar(0, 0, 200), -1);
            cv::circle(img, b, 3, cv::Scalar(0, 0, 200), -1);
        }
    }
}

void Openpose::GetSkeletonVectors(const std::vector<std::vector<cv::Point2f>> &keypoints,
                                  std::vector<std::vector<BodyStructure>> &skeleton) {
    skeleton.clear();
    for (size_t i = 0; i < keypoints.size(); i++) {
        std::vector<BodyStructure> unit;
        for (int n = 0; n < m_info.npairs; n++) {
            auto &points = keypoints[i];
            cv::Point2f a = points[POSE_PAIRS[m_info.midx][n][0]];
            cv::Point2f b = points[POSE_PAIRS[m_info.midx][n][1]];
            unit.push_back({a, b});
        }
        skeleton.push_back(unit);
    }
}

void Openpose::GetBodies(const float *heatmap_arr, const int W, const int H, const int channels,
                         const cv::Mat &img, std::vector<std::vector<cv::Point2f>> &keypoints) {
    if (m_info.detect_multi) {
        ResizeToNetInput(m_pbuf.heatmap_uptr, heatmap_arr, W, H, channels, net_width, net_height);

        int max_peaks = net_height - 1;
        const float &interThreshold =
            m_info.properties[(int)op::PoseProperty::ConnectInterThreshold];
        GetPeakMap(m_pbuf.peakmap_uptr, m_pbuf.kernel_uptr, m_info.multi_scalenet_to_output,
                   m_pbuf.heatmap_uptr, interThreshold, m_info.nparts, channels, max_peaks,
                   img.cols, img.rows, net_width, net_height);

        // Connect Body Parts
        op::Array<float> poseKeypoints;
        {
            op::Array<float> poseScores;
            const float &interMinAboveThreshold =
                m_info.properties[(int)op::PoseProperty::ConnectInterMinAboveThreshold];
            const float &interThreshold =
                m_info.properties[(int)op::PoseProperty::ConnectInterThreshold];
            const float &minSubsetCnt =
                m_info.properties[(int)op::PoseProperty::ConnectMinSubsetCnt];
            const float &minSubsetScore =
                m_info.properties[(int)op::PoseProperty::ConnectMinSubsetScore];
            op::Point<int> netSize{net_width, net_height};
            float *heatmap_ptr = m_pbuf.heatmap_uptr.get();
            float *peakmap_ptr = m_pbuf.peakmap_uptr.get();
            op::connectBodyPartsCpu(
                poseKeypoints, poseScores, heatmap_ptr, peakmap_ptr, m_info.pose_model, netSize,
                max_peaks, interMinAboveThreshold, interThreshold, minSubsetCnt, minSubsetScore,
                m_info.multi_scalenet_to_output, m_info.multi_maximize_positives);
        }

        // Push to BodyStructure
        const int numberPeopleDetected = poseKeypoints.getSize(0);
        const int numberBodyParts = poseKeypoints.getSize(1);
        for (int person = 0; person < numberPeopleDetected; person++) {
            std::vector<cv::Point2f> out;
            for (int part = 0; part < numberBodyParts; part++) {
                const int baseIndex = poseKeypoints.getSize(2) * (person * numberBodyParts + part);
                const float x = poseKeypoints[baseIndex];
                const float y = poseKeypoints[baseIndex + 1];
                // const float score = poseKeypoints[baseIndex + 2];
                cv::Point2f a(x, y);
                out.push_back(a);
            }
            keypoints.push_back(out);
        }
    } else {
        const float &interThreshold =
            m_info.properties[(int)op::PoseProperty::ConnectInterThreshold];
        // the result is an array of "heatmaps", the probability of a body part being in location
        // x,y find the position of the body parts
        std::vector<cv::Point> points(22);
        for (int n = 0; n < m_info.nparts; n++) {
            // Slice heatmap of corresponding body's part.
            const cv::Mat heatMap(H, W, CV_32FC1, (float *)(heatmap_arr + (n * (W * H))));
            // 1 maximum per heatmap
            cv::Point p(-1, -1), pm;
            double conf;
            cv::minMaxLoc(heatMap, 0, &conf, 0, &pm);
            if ((float)conf > interThreshold) p = pm;
            points[n] = p;
        }

        // connect body parts (and draw it) !
        float SX, SY;
        if (preserve_ratio) {
            SX = std::max(1.0 * img.rows / H, 1.0 * img.cols / W);
            SY = SX;
        } else {
            SX = float(img.cols) / W;
            SY = float(img.rows) / H;
        }

        std::vector<cv::Point2f> unit;
        for (int n = 0; n < m_info.npairs; n++) {
            // lookup 2 connected body/hand parts
            cv::Point2f a = points[n];

            // scale to image size
            a.x *= SX;
            a.y *= SY;

            unit.push_back(a);
        }
        keypoints.push_back(unit);
    }
}

void Openpose::ResizeToNetInput(unique_aptr<float> &heat_uptr, const float *heatmap_arr,
                                const int &origin_w, const int &origin_h, const int &origin_c,
                                const int &target_w, const int &target_h) {
    // TODO: FIXME: might break if input size changed
    if (!heat_uptr) {
        heat_uptr = allocate_aligned<float>(16, target_w * target_h * origin_c);
    }
    float *heatmap_ptr = heat_uptr.get();
    for (int i = 0; i < origin_c; i++) {
        const cv::Mat heatMap(cv::Size(origin_w, origin_h), CV_32FC1,
                              (float *)(heatmap_arr + (i * (origin_w * origin_h))));
        cv::Mat heap_resized_single(cv::Size(target_w, target_h), CV_32FC1,
                                    (float *)(heatmap_ptr + (i * (target_w * target_h))));
        cv::resize(heatMap, heap_resized_single, cv::Size(target_w, target_h), 0, 0,
                   CV_INTER_CUBIC);
    }
}

void Openpose::GetPeakMap(unique_aptr<float> &peakmap_uptr, unique_aptr<int> &kernel_uptr,
                          float &multi_scalenet_to_output, const unique_aptr<float> &heat_uptr,
                          const float &threshold, const int &nparts, const int &channels,
                          const int &max_peaks, const int &img_w, const int &img_h,
                          const int &heap_w, const int &heap_h) {
    int peak_c = nparts > 0 ? nparts : channels;
    int peak_h = max_peaks + 1;  // # maxPeaks + 1
    int peak_w = 3;              // X, Y, score

    if (!peakmap_uptr) {
        peakmap_uptr = allocate_aligned<float>(16, peak_w * peak_h * peak_c);
        kernel_uptr = allocate_aligned<int>(16, heap_w * heap_h * channels);
    }
    const float *heatmap_ptr = heat_uptr.get();
    int *kernel_ptr = kernel_uptr.get();
    float *peakmap_ptr = peakmap_uptr.get();

    const std::array<int, 4> targetSize({1, nparts, peak_h, peak_w});
    const std::array<int, 4> sourceSize({1, channels, peak_h, heap_w});

    const op::Point<int> inputDataSize({img_w, img_h});
    const op::Point<int> netOutputSize({heap_w, heap_h});
    const auto scaleProducerToNetInput = resizeGetScaleFactor(inputDataSize, netOutputSize);
    // FIXME: breaks when ratio is not kept
    const op::Point<int> netSize = {
        op::positiveIntRound(scaleProducerToNetInput * inputDataSize.x),
        op::positiveIntRound(scaleProducerToNetInput * inputDataSize.y)};
    multi_scalenet_to_output = {(float)resizeGetScaleFactor(netSize, inputDataSize)};
    const auto nmsOffset = float(0.5 / double(multi_scalenet_to_output));
    const op::Point<float> offset(nmsOffset, nmsOffset);
    op::nmsCpu(peakmap_ptr, kernel_ptr, heatmap_ptr, threshold, targetSize, sourceSize, offset);
}
}  // namespace vision
}  // namespace qnn