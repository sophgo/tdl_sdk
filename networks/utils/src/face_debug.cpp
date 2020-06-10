// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
#include <fstream>

#include "face_debug.hpp"

namespace qnn {
namespace vision {

void SaveCVImg(const char *name, const cv::Mat &src) { cv::imwrite(name, src); }

void DrawRectnSaveCVImg(const char *name, const cv::Mat &src,
                        std::vector<face_detect_rect_t> &rects) {
    cv::Mat temp;
    src.copyTo(temp);
    for (size_t i = 0; i < rects.size(); i++) {
        face_detect_rect_t &rect = rects[i];
        float x = rect.x1;
        float y = rect.y1;
        float w = rect.x2 - rect.x1 + 1;
        float h = rect.y2 - rect.y1 + 1;
        cv::rectangle(temp, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 1);
    }
    cv::imwrite(name, temp);
}

void SaveMat2Txt(const std::string name, const cv::Mat &src) {
    std::ofstream myfile(name.c_str());
    myfile << "Width: " << src.cols << "\n";
    myfile << "Height: " << src.rows << "\n";
    if (src.type() == CV_8UC3 || src.type() == CV_8SC3) {
        for (int j = 0; j < src.rows; j++) {
            for (int i = 0; i < src.cols; i++) {
                myfile << (int)src.at<cv::Vec3b>(j, i)[0] << "\n"
                       << (int)src.at<cv::Vec3b>(j, i)[1] << "\n"
                       << (int)src.at<cv::Vec3b>(j, i)[2] << "\n";
            }
        }
    } else if (src.type() == CV_8UC1 || src.type() == CV_8SC1) {
        for (int j = 0; j < src.rows; j++) {
            for (int i = 0; i < src.cols; i++) {
                myfile << (int)src.at<uchar>(j, i) << "\n";
            }
        }
    } else if (src.type() == CV_32FC3) {
        for (int j = 0; j < src.rows; j++) {
            for (int i = 0; i < src.cols; i++) {
                myfile << (float)src.at<cv::Vec3f>(j, i)[0] << "\n"
                       << (float)src.at<cv::Vec3f>(j, i)[1] << "\n"
                       << (float)src.at<cv::Vec3f>(j, i)[2] << "\n";
            }
        }
    }
    myfile.close();
}

void SaveTensor2Txt(const std::string name, InputTensor &in_tensor,
                    std::vector<OutputTensor> &out_tensors, bool is_app) {
    std::ofstream myfile;
    if (is_app)
        myfile.open(name.c_str(), std::ofstream::out | std::ofstream::app);
    else
        myfile.open(name.c_str());

    myfile << "Input tensor length: " << in_tensor.count << "\n";
    myfile << "Tensor shape (n, c, h, w) = (" << in_tensor.shape.n << ", " << in_tensor.shape.c
           << ", " << in_tensor.shape.h << ", " << in_tensor.shape.w << ")\n";
    for (int i = 0; i < in_tensor.count; i++) {
        myfile << (int)in_tensor.data[i] << "\n";
    }
    myfile << "============================================\n";
    myfile << "\n";
    myfile << "============================================\n";

    for (size_t j = 0; j < out_tensors.size(); j++) {
        auto &out_tensor = out_tensors[j];
        myfile << "Output tensor length: " << out_tensor.count << "\n";
        myfile << "Tensor shape (n, c, h, w) = (" << out_tensor.shape.n << ", "
               << out_tensor.shape.c << ", " << out_tensor.shape.h << ", " << out_tensor.shape.w
               << ")\n";
        for (int i = 0; i < out_tensor.count; i++) {
            myfile << (int)out_tensor.q_data[i] << "\n";
        }
        myfile << "============================================\n";
        myfile << "\n";
        myfile << "============================================\n";
    }

    myfile.close();
}

void SaveFDInfo2Txt(const std::string name, std::vector<face_info_regression_t> &infos,
                    bool is_app) {
    std::ofstream myfile;
    if (is_app)
        myfile.open(name.c_str(), std::ofstream::out | std::ofstream::app);
    else
        myfile.open(name.c_str());
    myfile << "Vector size: " << infos.size() << "\n";
    for (size_t i = 0; i < infos.size(); i++) {
        face_detect_rect_t &rect = infos[i].bbox;
        myfile << "[Idx] " << i << "\n";
        myfile << "    Position: (x1, y1), (x2, y2) = (" << rect.x1 << ", " << rect.y1 << "), ("
               << rect.x2 << ", " << rect.y2 << ")\n";
        std::array<float, 4> &reg = infos[i].regression;
        myfile << "    Score: " << rect.score << "\n";
        myfile << "    Regression: (x1, y1), (x2, y2) = (" << reg[0] << ", " << reg[1] << "), ("
               << reg[2] << ", " << reg[3] << ")\n\n";
    }

    myfile.close();
}

}  // namespace vision
}  // namespace qnn