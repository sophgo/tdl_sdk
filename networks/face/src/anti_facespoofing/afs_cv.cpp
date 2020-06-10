// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>

#include "anti_facespoofing/afs_cv.hpp"
#include "utils/function_tracer.h"
#include <fstream>
#define PROCESS_DOWN_SCALE 4

namespace qnn {
namespace vision {

AntiFaceSpoofingCVBG::AntiFaceSpoofingCVBG() {}

void AntiFaceSpoofingCVBG::FrameDiff(const cv::Mat &img, const float &lr_rate, cv::Mat &blur_img,
                                     cv::Mat &morph_img) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray_img, blur_img, cv::Size(5, 5), 1.5);

    if (m_bparam.prev_frame.empty()) {
        m_bparam.prev_frame = blur_img.clone();
    }
    cv::Mat frame_delta, thresh_img;
    cv::absdiff(m_bparam.prev_frame, blur_img, frame_delta);
    cv::threshold(frame_delta, thresh_img, 10, 255, cv::THRESH_BINARY);
    cv::erode(thresh_img, morph_img, m_bparam.kerenl_erode);
    cv::dilate(morph_img, morph_img, m_bparam.kerenl_dilate, cv::Point(-1, -1), 3);
    if (lr_rate > 0) {
        m_bparam.prev_frame = blur_img.clone();
    }
    return;
}

int AntiFaceSpoofingCVBG::HeadForegroundDetect(const cv::Mat &ds_img, const cv::Mat &morph_img,
                                               const std::vector<cv::Rect> &ds_rect) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    float ratio_th = 0.6;
    int padd_large = 2;
    int padd_small = 4;
    int liveness = 1;
    for (size_t i = 0; i < ds_rect.size(); i++) {
        const int &x = ds_rect[i].x;
        const int &y = ds_rect[i].y;
        const int &w = ds_rect[i].width;
        const int &h = ds_rect[i].height;
        int &&x2 = x + w;
        int &&y2 = y + h;
        cv::Point2i t_pt1(int(x - w / padd_small), int(y - h / padd_large));
        cv::Point2i t_pt2(int(x2 + w / padd_small), int(y - h / padd_small));
        cv::Point2i l_pt1(int(x - w / padd_large), int(y));
        cv::Point2i l_pt2(int(x - w / padd_small), int(y2 - h / 2));
        cv::Point2i r_pt1(int(x2 + w / padd_large), int(y));
        cv::Point2i r_pt2(int(x2 + w / padd_small), int(y2 - h / 2));
        float area = (l_pt2.x - l_pt1.x) * (l_pt2.y - l_pt1.y);
        cv::Mat l_rect = cv::Mat::zeros(ds_img.size(), CV_8UC1);
        cv::Mat r_rect = cv::Mat::zeros(ds_img.size(), CV_8UC1);
        cv::Mat t_rect = cv::Mat::zeros(ds_img.size(), CV_8UC1);
        cv::rectangle(l_rect, cv::Rect(l_pt1, l_pt2), 255, -1);
        cv::rectangle(r_rect, cv::Rect(r_pt1, r_pt2), 255, -1);
        cv::rectangle(t_rect, cv::Rect(t_pt1, t_pt2), 255, -1);
        cv::Mat l_motion, r_motion, t_motion;
        cv::bitwise_and(morph_img, morph_img, l_motion, l_rect);
        cv::bitwise_and(morph_img, morph_img, r_motion, r_rect);
        cv::bitwise_and(morph_img, morph_img, t_motion, t_rect);
        float l_motion_ratio = cv::countNonZero(l_motion) / area;
        float r_motion_ratio = cv::countNonZero(r_motion) / area;
        float t_motion_ratio = cv::countNonZero(t_motion) / area;

        // #print("{} {} {}".format(l_motion_ratio, r_motion_ratio, t_motion_ratio))
        // if (1) {
        //     cv::Mat debug_img = ds_img.clone();
        //     cv::rectangle(debug_img, cv::Rect(l_pt1, l_pt2), cv::Scalar(0, 0, 255), 2);
        //     cv::rectangle(debug_img, cv::Rect(r_pt1, r_pt2), cv::Scalar(0, 255, 0), 2);
        //     cv::rectangle(debug_img, cv::Rect(t_pt1, t_pt2), cv::Scalar(255, 0, 0), 2);
        //     cv::imshow("HeadForegroundDetect", debug_img);
        // }

        int score = 0;
        if (l_motion_ratio > ratio_th) score += 1;
        if (r_motion_ratio > ratio_th) score += 1;
        if (t_motion_ratio > ratio_th) score += 1;
        if (score > 1) liveness = 0;
        break;  // process the biggest face
    }
    return liveness;
}
int AntiFaceSpoofingCVBG::FrameRectDetect(const cv::Mat &ds_img, const cv::Mat &blur_img,
                                          const cv::Mat &morph_img,
                                          const std::vector<cv::Rect> &ds_rect) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    int liveness = 1;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(morph_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Rect largest_rect(0, 0, 0, 0);
    int largest_area = -1;
    for (size_t i = 0; i < contours.size(); i++) {
        float area = cv::contourArea(contours[i]);
        if (area < 40 * 40) {
            continue;
        } else if (largest_area < area) {
            largest_area = area;
            largest_rect = cv::boundingRect(contours[i]);
        }
    }
    cv::Mat rect_img = cv::Mat::zeros(ds_img.size(), CV_8UC1);
    cv::Mat edge_img;
    cv::rectangle(rect_img, largest_rect, 255, -1);
    cv::Canny(blur_img, edge_img, 30, 100);
    cv::bitwise_and(edge_img, edge_img, edge_img, rect_img);
    cv::findContours(edge_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::sort(contours.begin(), contours.end(),
              [](auto &a, auto &b) { return fabs(cv::contourArea(a)) > fabs(cv::contourArea(b)); });

    for (size_t i = 0; i < contours.size(); i++) {
        float area = cv::contourArea(contours[i]);
        if (area < 30 * 30) {
            continue;
        }
        cv::Rect cont_rect = cv::boundingRect(contours[i]);
        for (size_t j = 0; j < 1; j++) {
            const cv::Rect &rect = ds_rect[j];
            int &&x2 = rect.width + rect.x;
            int &&y2 = rect.height + rect.y;
            int &&cx2 = cont_rect.width + cont_rect.x;
            int &&cy2 = cont_rect.height + cont_rect.y;
            // rect size bigger and face located in rect
            if (cont_rect.width > rect.width && cont_rect.height > rect.height &&
                rect.x > cont_rect.x && rect.y > cont_rect.y && x2 < cx2 && y2 < cy2) {
                liveness = 0;
                // if (1) {
                //     cv::Mat debug_img = ds_img.clone();
                //     cv::rectangle(debug_img, cont_rect, cv::Scalar(0, 0, 255), 1);
                //     cv::rectangle(debug_img, rect, cv::Scalar(255, 0, 0), 1);
                //     cv::imshow("FrameRectDetect", debug_img);
                // }
            }
        }
    }
    // cv::imshow("Edge", edge_img);
    return liveness;
}

int AntiFaceSpoofingCVBG::BGDetect(const cv::Mat &image, const std::vector<cv::Rect> &face_rect) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    cv::Mat ds_img;
    if (PROCESS_DOWN_SCALE > 1) {
        cv::resize(image, ds_img,
                   cv::Size(image.cols / PROCESS_DOWN_SCALE, image.rows / PROCESS_DOWN_SCALE), 0, 0,
                   cv::INTER_CUBIC);
    } else {
        ds_img = image.clone();
    }

    std::vector<cv::Rect> ds_rect;

    for (size_t i = 0; i < face_rect.size(); i++) {
        const cv::Rect &rect = face_rect[i];
        ds_rect.push_back(cv::Rect(rect.x / PROCESS_DOWN_SCALE, rect.y / PROCESS_DOWN_SCALE,
                                   rect.width / PROCESS_DOWN_SCALE,
                                   rect.height / PROCESS_DOWN_SCALE));
    }

    float lr_rate = m_bparam.skip ? 0.5 : (m_bparam.motion ? 0 : 0.1);
    cv::Mat blur_img, morph_img;
    FrameDiff(ds_img, lr_rate, blur_img, morph_img);
    m_bparam.skip = false;
    if (m_bparam.bg_update_state == 0) {
        int area = cv::countNonZero(morph_img);
        if (area > 0.05 * ds_img.cols * ds_img.rows) {
            m_bparam.bg_start_freeze_time = std::chrono::steady_clock::now();
            m_bparam.bg_update_state = 1;
            m_bparam.motion = true;
        }
    }

    if (ds_rect.size() > 0) {
        m_bparam.face_vanish_count = 0;
        // if (m_bparam.bg_update_state == 1) {
        //     m_bparam.bg_start_freeze_time = std::chrono::system_clock::zeros();
        // }
        m_bparam.motion = true;
    } else {
        m_bparam.face_vanish_count++;
        if (m_bparam.face_vanish_count > m_bparam.face_vanish_threshold) {
            if (m_bparam.bg_update_state == 1) {
                auto fin = std::chrono::steady_clock::now();
                std::chrono::duration<float> dur = fin - m_bparam.bg_start_freeze_time;
                if (dur.count() > m_bparam.bg_resume_update_time) {
                    m_bparam.motion = false;
                    m_bparam.face_vanish_count = 0;
                    m_bparam.bg_update_state = 0;
                }
            } else if (m_bparam.bg_update_state == 2) {
                m_bparam.motion = false;
                m_bparam.face_vanish_count = 0;
                m_bparam.bg_update_state = 0;
            }
        }
    }

    int motion_liveness = HeadForegroundDetect(ds_img, morph_img, ds_rect);
    int rect_line_liveness = 1;
    if (motion_liveness) {
        rect_line_liveness = FrameRectDetect(ds_img, blur_img, morph_img, ds_rect);
    }
    int liveness = motion_liveness * rect_line_liveness;
    return liveness;
}

bool AntiFaceSpoofingCVSharpness::Detect(cv::Mat &img) {
    m_sharpness_val = GetSharpnessLevel(img);
    LOGI << "Sharpness value " << m_sharpness_val << " threshold " << m_sharpness_threshold;
    return (m_sharpness_val > m_sharpness_threshold) ? true : false;
}

void AntiFaceSpoofingCVSharpness::SetThreshold(uint value) { m_sharpness_threshold = value; }

const uint AntiFaceSpoofingCVSharpness::GetThreshold() { return m_sharpness_threshold; }

const uint AntiFaceSpoofingCVSharpness::GetSharpnessVal() { return m_sharpness_val; }

const uint AntiFaceSpoofingCVSharpness::GetSharpnessLevel(cv::Mat &img) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    cv::Mat laplacian_img;
    cv::Laplacian(img, laplacian_img, CV_32F);
    cv::Scalar mean, stddev;  // 0:1st channel, 1:2nd channel and 2:3rd channel
    cv::meanStdDev(laplacian_img, mean, stddev, cv::Mat());
    uint sharpess_val = stddev.val[0] * stddev.val[0];
    return sharpess_val;
}

AntiFaceSpoofingLBPSVM::~AntiFaceSpoofingLBPSVM() {
    // LOGI << "~AntiFaceSpoofingLBPSVM(), m_pMean: " << m_pMean
    // << ", m_pStd: " << m_pStd;
    if (m_pLinear != nullptr) {
        delete m_pLinear;
        m_pLinear = nullptr;
    }
    if (m_pMean != nullptr) {
        delete[] m_pMean;
        m_pMean = nullptr;
    }
    if (m_pStd != nullptr) {
        delete[] m_pStd;
        m_pStd = nullptr;
    }
}

AntiFaceSpoofingLBPSVM::AntiFaceSpoofingLBPSVM(const std::string& model_path,
                                               const std::string& mean_path,
                                               const std::string& std_path) {
    if (!m_bModelLoaded) {
        LoadLBPSVMModel(model_path, mean_path, std_path);
    }
}

void AntiFaceSpoofingLBPSVM::LoadLBPSVMModel(const std::string& model_path,
                                             const std::string& mean_path,
                                             const std::string& std_path) {
    if (!m_bModelLoaded) {
        m_pLinear = new LibLinear;
        m_pLinear->load_model(model_path);
        m_dim_feature = GetDimFeat(m_radius, m_neighbors);

        std::ifstream fmean;
        std::ifstream fstd;
        fmean.open(mean_path);
        if (!fmean) {
            LOGI << "Open SVM mean data failed ! " << mean_path;
        }
        fstd.open(std_path);
        if (!fstd) {
            LOGI << "Open SVM std data failed ! " << std_path;
        }

        uint t;
        fmean.read(reinterpret_cast<char*>(&t), sizeof(uint));
        if (t != m_dim_feature) {
            LOGI << "Feature Dimension is incorrect ! " << t
                        << ", expected: " << m_dim_feature;
            fmean.close();
            return;
        }
        m_pMean = new float[m_dim_feature];
        fmean.read(reinterpret_cast<char*>(m_pMean), sizeof(float)*m_dim_feature);
        fmean.close();

        fstd.read(reinterpret_cast<char*>(&t), sizeof(uint));
        if (t != m_dim_feature) {
            LOGI << "Feature Dimension is incorrect ! " << t
                        << ", expected: " << m_dim_feature;
            fstd.close();
            return;
        }
        m_pStd = new float[m_dim_feature];
        fstd.read(reinterpret_cast<char*>(m_pStd), sizeof(float)*m_dim_feature);
        fstd.close();

        m_bModelLoaded = true;
    }
}

bool AntiFaceSpoofingLBPSVM::Detect(const cv::Mat &image, const face_info_t &face_info) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);

    if (!m_bModelLoaded) {
        LOGE << "LBP SVM model didn't loaded, please call LoadLBPSVMModel() before call detect()";
    }

    cv::Mat aligned(112, 112, image.type());
    if (face_align(image, aligned, face_info, 112, 112) != qnn::RET_SUCCESS) {
        LOGI << "face_align failed";
    }

#if 0 // Test usage
    cv::Mat tmp;
    tmp = cv::imread("real_3.png");
    cv::Mat croppedImg;
    int fx = 38, fy = 50, fw = 36, fh = 42;
	tmp(cv::Rect(fx, fy, fw, fh)).copyTo(croppedImg);
    cv::Mat one_feat = GetFeatureVector(croppedImg);
#else
    cv::Mat croppedImg; // Latest SVM use the whoe 112x112 images
	// int fx = 38, fy = 50, fw = 36, fh = 42;
	// aligned(cv::Rect(fx, fy, fw, fh)).copyTo(croppedImg);
    cv::Mat one_feat = GetFeatureVector(aligned);
#endif

    double result = m_pLinear->predict_values((float*)one_feat.data, one_feat.cols, &m_result);
    LOGI << "LBP SVM Predict: " << m_result << ", Result: " << result;
    return (m_result > m_lbpsvm_threshold) ? true : false;
}

cv::Mat AntiFaceSpoofingLBPSVM::GetFeatureVector(cv::Mat crop_rgb_face, bool bNorm) {
    std::vector<cv::Mat> hsv_planes, ycc_planes;
	cv::Mat hsv, ycc, img_lbp;

	cv::cvtColor(crop_rgb_face, hsv, cv::COLOR_BGR2HSV);
	cv::split(hsv, hsv_planes);
	cv::cvtColor(crop_rgb_face, ycc, cv::COLOR_BGR2YCrCb);
	cv::split(ycc, ycc_planes);

	int radius = 3 * 3;
	int n_points = 8 * radius;
	GetUniformPatternLBPFeature(ycc_planes[0], img_lbp, radius, n_points);
	//cv::imwrite("resultUniform.jpg", img_lbp);
	cv::Mat hist_cell_y = GetLocalRegionLBPH(img_lbp, 0, n_points, false);

	img_lbp.release();
	GetUniformPatternLBPFeature(ycc_planes[1], img_lbp, radius, n_points);
	cv::Mat hist_cell_cr = GetLocalRegionLBPH(img_lbp, 0, n_points, false);

	img_lbp.release();
	GetUniformPatternLBPFeature(ycc_planes[2], img_lbp, radius, n_points);
	cv::Mat hist_cell_cb = GetLocalRegionLBPH(img_lbp, 0, n_points, false);

	// hsv
	img_lbp.release();
	GetUniformPatternLBPFeature(hsv_planes[0], img_lbp, radius, n_points);
	cv::Mat hist_cell_h = GetLocalRegionLBPH(img_lbp, 0, n_points, false);

	img_lbp.release();
	GetUniformPatternLBPFeature(hsv_planes[1], img_lbp, radius, n_points);
	cv::Mat hist_cell_s = GetLocalRegionLBPH(img_lbp, 0, n_points, false);

	img_lbp.release();
	GetUniformPatternLBPFeature(hsv_planes[2], img_lbp, radius, n_points);
	cv::Mat hist_cell_v = GetLocalRegionLBPH(img_lbp, 0, n_points, false);
	cv::Mat final_feature;

	cv::Mat matArray[] = { hist_cell_y, hist_cell_cr, hist_cell_cb, hist_cell_h, hist_cell_s, hist_cell_v };
	cv::hconcat(matArray, 6, final_feature);
	cv::normalize(final_feature, final_feature);

    if (bNorm) {
        int dim_feat = final_feature.cols;

        for (int j = 0; j < dim_feat; j++) {
            final_feature.at<float>(0, j) -= m_pMean[j];
            final_feature.at<float>(0, j) /= (m_pStd[j] + 1e-7);
        }
    }

	return final_feature;
}

void AntiFaceSpoofingLBPSVM::GetCircularLBPFeatureOptimization(cv::InputArray _src, cv::OutputArray _dst, int radius, int neighbors) {
    cv::Mat src = _src.getMat();

	_dst.create(src.rows - 2 * radius, src.cols - 2 * radius, CV_8UC1);
	cv::Mat dst = _dst.getMat();
	dst.setTo(0);
	for (int k = 0; k<neighbors; k++) {

		float rx = static_cast<float>(radius * cos(2.0 * CV_PI * k / neighbors));
		float ry = -static_cast<float>(radius * sin(2.0 * CV_PI * k / neighbors));

		int x1 = static_cast<int>(floor(rx));
		int x2 = static_cast<int>(ceil(rx));
		int y1 = static_cast<int>(floor(ry));
		int y2 = static_cast<int>(ceil(ry));

		float tx = rx - x1;
		float ty = ry - y1;

		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx * (1 - ty);
		float w3 = (1 - tx) * ty;
		float w4 = tx * ty;

		for (int i = radius; i<src.rows - radius; i++) {
			for (int j = radius; j<src.cols - radius; j++) {

				uchar center = src.at<uchar>(i, j);

				float neighbor = src.at<uchar>(i + x1, j + y1) * w1 + src.at<uchar>(i + x1, j + y2) *w2 \
					+ src.at<uchar>(i + x2, j + y1) * w3 + src.at<uchar>(i + x2, j + y2) *w4;

				dst.at<uchar>(i - radius, j - radius) |= (neighbor>center) << (neighbors - k - 1);
			}
		}
	}
}

static int getHopTimes(int n)
{
	int count = 0;
	std::bitset<8> binaryCode = n;
	for (int i = 0; i < 8; i++) {
		if (binaryCode[i] != binaryCode[(i + 1) % 8]) {
			count++;
		}
	}
	return count;
}

void AntiFaceSpoofingLBPSVM::GetUniformPatternLBPFeature(cv::InputArray _src, cv::OutputArray _dst, int radius, int neighbors) {
    cv::Mat src = _src.getMat();

	_dst.create(src.rows - 2 * radius, src.cols - 2 * radius, CV_8UC1);
	cv::Mat dst = _dst.getMat();
	dst.setTo(0);

	uchar temp = 1;
	uchar table[256] = { 0 };
	for (int i = 0; i < 256; i++)
	{
		if (getHopTimes(i) < 3)
		{
			table[i] = temp;
			temp++;
		}
	}

	bool flag = false;

	for (int k = 0; k < neighbors; k++) {
		if (k == neighbors - 1)	{
			flag = true;
		}

		float rx = static_cast<float>(radius * cos(2.0 * CV_PI * k / neighbors));
		float ry = -static_cast<float>(radius * sin(2.0 * CV_PI * k / neighbors));

		int x1 = static_cast<int>(floor(rx));
		int x2 = static_cast<int>(ceil(rx));
		int y1 = static_cast<int>(floor(ry));
		int y2 = static_cast<int>(ceil(ry));

		float tx = rx - x1;
		float ty = ry - y1;

		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx * (1 - ty);
		float w3 = (1 - tx) * ty;
		float w4 = tx * ty;

		for (int i = radius; i<src.rows - radius; i++) {
			for (int j = radius; j<src.cols - radius; j++) {
				uchar center = src.at<uchar>(i, j);

				float neighbor = src.at<uchar>(i + x1, j + y1) * w1 + src.at<uchar>(i + x1, j + y2) *w2 \
					+ src.at<uchar>(i + x2, j + y1) * w3 + src.at<uchar>(i + x2, j + y2) *w4;

				dst.at<uchar>(i - radius, j - radius) |= (neighbor>center) << (neighbors - k - 1);

				if (flag) {
					dst.at<uchar>(i - radius, j - radius) = table[dst.at<uchar>(i - radius, j - radius)];
				}
			}
		}
	}
}
cv::Mat AntiFaceSpoofingLBPSVM::GetLocalRegionLBPH(const cv::Mat& src, int minValue, int maxValue, bool normed) {

	cv::Mat result;

	int histSize = maxValue - minValue + 2; // default: maxValue - minValue + 1;

	float range[] = { static_cast<float>(minValue),static_cast<float>(maxValue + 1) }; // default: maxValue + 1
	const float* ranges = { range };

	calcHist(&src, 1, 0, cv::Mat(), result, 1, &histSize, &ranges, true, false);

	if (normed) {
		result /= (int)src.total();
	}

	return result.reshape(1, 1);
}
cv::Mat AntiFaceSpoofingLBPSVM::GetLBPH(cv::InputArray _src, int numPatterns, int grid_x, int grid_y, bool normed) {
    cv::Mat src = _src.getMat();
	int width = src.cols / grid_x;
	int height = src.rows / grid_y;

	cv::Mat result = cv::Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
	if (src.empty())
	{
		return result.reshape(1, 1);
	}
	int resultRowIndex = 0;

	for (int i = 0; i<grid_x; i++)
	{
		for (int j = 0; j<grid_y; j++)
		{

			cv::Mat src_cell = cv::Mat(src, cv::Range(i*height, (i + 1)*height), cv::Range(j*width, (j + 1)*width));

			cv::Mat hist_cell = GetLocalRegionLBPH(src_cell, 0, (numPatterns - 1), true);

			cv::Mat rowResult = result.row(resultRowIndex);
			hist_cell.reshape(1, 1).convertTo(rowResult, CV_32FC1);
			resultRowIndex++;
		}
	}
	return result.reshape(1, 1);
}
int AntiFaceSpoofingLBPSVM::GetDimFeat(int radius, int neighbors) {
    return (radius*neighbors + 2) * 3 * 2;
}

}  // namespace vision
}  // namespace qnn