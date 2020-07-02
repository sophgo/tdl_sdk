#pragma once

#include "generic_image.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

class cvmat_bgr_image_t {
private:
    cv::Mat mat_;

public:
    cvmat_bgr_image_t() {}
    cvmat_bgr_image_t(const cvmat_bgr_image_t &) = default;
    cvmat_bgr_image_t(cvmat_bgr_image_t &&) = default;
    cvmat_bgr_image_t(cv::Mat &&mat) {
        *this = std::move(mat);
    }
    cvmat_bgr_image_t(const cv::Mat &mat) {
        *this = mat;
    }
    cvmat_bgr_image_t(generic_image_t &&img) {
        switch (img.metadata.format) {
        case generic_image_t::NONE:
            break;
        case generic_image_t::BGR:
            *this = cv::Mat(img.metadata.bgr_info.height, img.metadata.bgr_info.width, CV_8UC3, (void *)img.data.data());
            break;
        case generic_image_t::JPG:
            *this = cv::imdecode(cv::Mat(1, img.data.size(), CV_8UC1, img.data.data()), cv::IMREAD_COLOR);
            break;
        default:
            throw std::invalid_argument("invalid image format");
        }
    }
    cvmat_bgr_image_t &operator=(cv::Mat &&mat) {
        if (mat.type() != CV_8UC3) {
            mat.convertTo(mat, CV_8UC3);
        }

        if (mat.u == nullptr || mat.u->refcount == 0) {
            mat.copyTo(mat_);
        } else {
            mat_ = mat;
        }

        return *this;
    }
    cvmat_bgr_image_t &operator=(const cv::Mat &mat) {
        return (*this = cv::Mat(mat));
    }
    cvmat_bgr_image_t &operator=(const cvmat_bgr_image_t &) = default;
    cvmat_bgr_image_t &operator=(cvmat_bgr_image_t &&) = default;

    operator const cv::Mat &() const {
        return mat_;
    }
    operator cv::Mat &() {
        return mat_;
    }

    operator generic_image_t() const {
        cv::Mat tmp;
        if (mat_.isContinuous()) {
            tmp = mat_;
        } else {
            mat_.copyTo(tmp);
        }

        memory::Bytes content(std::move(tmp), tmp.ptr(0), tmp.elemSize() * tmp.total());
        generic_image_t::metadata_t metadata;

        metadata.format = generic_image_t::BGR;
        metadata.bgr_info.width = mat_.size[1];
        metadata.bgr_info.height = mat_.size[0];

        return generic_image_t(metadata, std::move(content));
    }

    generic_image_t to_format(generic_image_t::FORMAT format) {
        switch (format) {
        case generic_image_t::BGR:
            return generic_image_t(*this);
        case generic_image_t::JPG:
            {
                generic_image_t::metadata_t metadata;
                metadata.format = generic_image_t::JPG;
                std::vector<unsigned char> buf;
                cv::imencode(".jpg", mat_, buf);
                return generic_image_t(metadata, memory::Bytes(std::move(buf)));
            }
        default:
            throw std::invalid_argument("invalid image format");
        }
    }

    bool empty() const {
        return mat_.empty();
    }
};

namespace memory {

template <>
class Encoding<cvmat_bgr_image_t> {
public:
    static std::list<memory::Bytes> encode(cvmat_bgr_image_t &&data) {
        return memory::Encoding<generic_image_t>::encode(generic_image_t(data));
    }

    static void decode(std::list<memory::Bytes> &list, cvmat_bgr_image_t &output) {
        generic_image_t tmp_image;
        memory::Encoding<generic_image_t>::decode(list, tmp_image);
        output = cvmat_bgr_image_t(std::move(tmp_image));
    }
};

} // namespace memory
