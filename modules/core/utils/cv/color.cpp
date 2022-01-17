#include "color.hpp"
#include "types_c.h"

#define EXT_FUNCTION 0

// constants for conversion from/to RGB and Gray, YUV, YCrCb according to BT.601
// L109
const float B2YF = 0.114f;
const float G2YF = 0.587f;
const float R2YF = 0.299f;

// L236
template <typename _Tp>
struct ColorChannel {
  typedef float worktype_f;
  static _Tp max() { return std::numeric_limits<_Tp>::max(); }
  static _Tp half() { return (_Tp)(max() / 2 + 1); }
};

///////////////////////////// Top-level template function ////////////////////////////////
// L260
template <typename Cvt>
class CvtColorLoop_Invoker : public cv::ParallelLoopBody {
  typedef typename Cvt::channel_type _Tp;

 public:
  CvtColorLoop_Invoker(const uchar* src_data_, size_t src_step_, uchar* dst_data_, size_t dst_step_,
                       int width_, const Cvt& _cvt)
      : ParallelLoopBody(),
        src_data(src_data_),
        src_step(src_step_),
        dst_data(dst_data_),
        dst_step(dst_step_),
        width(width_),
        cvt(_cvt) {}

  virtual void operator()(const cv::Range& range) const {
    const uchar* yS = src_data + static_cast<size_t>(range.start) * src_step;
    uchar* yD = dst_data + static_cast<size_t>(range.start) * dst_step;

    for (int i = range.start; i < range.end; ++i, yS += src_step, yD += dst_step)
      cvt(reinterpret_cast<const _Tp*>(yS), reinterpret_cast<_Tp*>(yD), width);
  }

 private:
  const uchar* src_data;
  size_t src_step;
  uchar* dst_data;
  size_t dst_step;
  int width;
  const Cvt& cvt;

  const CvtColorLoop_Invoker& operator=(const CvtColorLoop_Invoker&);
};

// L292
template <typename Cvt>
void CvtColorLoop(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                  int width, int height, const Cvt& cvt) {
  cv::parallel_for_(cv::Range(0, height),
                    CvtColorLoop_Invoker<Cvt>(src_data, src_step, dst_data, dst_step, width, cvt),
                    (width * height) / static_cast<double>(1 << 16));
}

////////////////// Various 3/4-channel to 3/4-channel RGB transformations /////////////////
// L652
template <typename _Tp>
struct RGB2RGB {
  typedef _Tp channel_type;

  RGB2RGB(int _srccn, int _dstcn, int _blueIdx) : srccn(_srccn), dstcn(_dstcn), blueIdx(_blueIdx) {}
  void operator()(const _Tp* src, _Tp* dst, int n) const {
    int scn = srccn, dcn = dstcn, bidx = blueIdx;
    if (dcn == 3) {
      n *= 3;
      for (int i = 0; i < n; i += 3, src += scn) {
        _Tp t0 = src[bidx], t1 = src[1], t2 = src[bidx ^ 2];
        dst[i] = t0;
        dst[i + 1] = t1;
        dst[i + 2] = t2;
      }
    } else if (scn == 3) {
      n *= 3;
      _Tp alpha = ColorChannel<_Tp>::max();
      for (int i = 0; i < n; i += 3, dst += 4) {
        _Tp t0 = src[i], t1 = src[i + 1], t2 = src[i + 2];
        dst[bidx] = t0;
        dst[1] = t1;
        dst[bidx ^ 2] = t2;
        dst[3] = alpha;
      }
    } else {
      n *= 4;
      for (int i = 0; i < n; i += 4) {
        _Tp t0 = src[i], t1 = src[i + 1], t2 = src[i + 2], t3 = src[i + 3];
        dst[i] = t2;
        dst[i + 1] = t1;
        dst[i + 2] = t0;
        dst[i + 3] = t3;
      }
    }
  }

  int srccn, dstcn, blueIdx;
};

///////////////////////////////// Color to/from Grayscale ////////////////////////////////
// L1037
template <typename _Tp>
struct Gray2RGB {
  typedef _Tp channel_type;

  Gray2RGB(int _dstcn) : dstcn(_dstcn) {}
  void operator()(const _Tp* src, _Tp* dst, int n) const {
    if (dstcn == 3)
      for (int i = 0; i < n; i++, dst += 3) {
        dst[0] = dst[1] = dst[2] = src[i];
      }
    else {
      _Tp alpha = ColorChannel<_Tp>::max();
      for (int i = 0; i < n; i++, dst += 4) {
        dst[0] = dst[1] = dst[2] = src[i];
        dst[3] = alpha;
      }
    }
  }

  int dstcn;
};

// L1171
#undef R2Y
#undef G2Y
#undef B2Y
enum {
  yuv_shift = 14,
  xyz_shift = 12,
  R2Y = 4899,  // == R2YF*16384
  G2Y = 9617,  // == G2YF*16384
  B2Y = 1868,  // == B2YF*16384
  BLOCK_SIZE = 256
};

// L1343
template <typename _Tp>
struct RGB2Gray {
  typedef _Tp channel_type;

  RGB2Gray(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn) {
    static const float coeffs0[] = {R2YF, G2YF, B2YF};
    memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 3 * sizeof(coeffs[0]));
    if (blueIdx == 0) std::swap(coeffs[0], coeffs[2]);
  }

  void operator()(const _Tp* src, _Tp* dst, int n) const {
    int scn = srccn;
    float cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];
    for (int i = 0; i < n; i++, src += scn)
      dst[i] = cv::saturate_cast<_Tp>(src[0] * cb + src[1] * cg + src[2] * cr);
  }
  int srccn;
  float coeffs[3];
};

// L1366
template <>
struct RGB2Gray<uchar> {
  typedef uchar channel_type;

  RGB2Gray(int _srccn, int blueIdx, const int* coeffs) : srccn(_srccn) {
    const int coeffs0[] = {R2Y, G2Y, B2Y};
    if (!coeffs) coeffs = coeffs0;

    int b = 0, g = 0, r = (1 << (yuv_shift - 1));
    int db = coeffs[blueIdx ^ 2], dg = coeffs[1], dr = coeffs[blueIdx];

    for (int i = 0; i < 256; i++, b += db, g += dg, r += dr) {
      tab[i] = b;
      tab[i + 256] = g;
      tab[i + 512] = r;
    }
  }
  void operator()(const uchar* src, uchar* dst, int n) const {
    int scn = srccn;
    const int* _tab = tab;
    for (int i = 0; i < n; i++, src += scn)
      dst[i] = (uchar)((_tab[src[0]] + _tab[src[1] + 256] + _tab[src[2] + 512]) >> yuv_shift);
  }
  int srccn;
  int tab[256 * 3];
};

// TODO: RGB2Gray CV_NEON version

// 8u, 16u, 32f
// L8643
void cvtBGRtoBGR(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                 int width, int height, int depth, int scn, int dcn, bool swapBlue) {
  int blueIdx = swapBlue ? 2 : 0;
  if (depth == CV_8U) {
    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height,
                 RGB2RGB<uchar>(scn, dcn, blueIdx));
  } else if (depth == CV_16U) {
    CV_Assert(0);
  } else {
    CV_Assert(0);
  }
}

// 8u, 16u, 32f
// L8804
void cvtBGRtoGray(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                  int width, int height, int depth, int scn, bool swapBlue) {
  int blueIdx = swapBlue ? 2 : 0;
  if (depth == CV_8U) {
    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height,
                 RGB2Gray<uchar>(scn, blueIdx, 0));
  } else if (depth == CV_16U) {
    CV_Assert(0);
  } else {
    CV_Assert(0);
  }
}

// 8u, 16u, 32f
// L8853
void cvtGraytoBGR(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                  int width, int height, int depth, int dcn) {
  if (depth == CV_8U) {
    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Gray2RGB<uchar>(dcn));
  } else if (depth == CV_16U) {
    CV_Assert(0);
  } else {
    CV_Assert(0);
  }
}

//
// Helper functions
//

// clang-format off
// L9652
inline bool swapBlue(int code) {
  switch (code)
  {
  case CV_BGR2BGRA: case CV_BGRA2BGR:
  case CV_BGR2BGR565: case CV_BGR2BGR555: case CV_BGRA2BGR565: case CV_BGRA2BGR555:
  case CV_BGR5652BGR: case CV_BGR5552BGR: case CV_BGR5652BGRA: case CV_BGR5552BGRA:
  case CV_BGR2GRAY: case CV_BGRA2GRAY:
  case CV_BGR2YCrCb: case CV_BGR2YUV:
  case CV_YCrCb2BGR: case CV_YUV2BGR:
  case CV_BGR2XYZ: case CV_XYZ2BGR:
  case CV_BGR2HSV: case CV_BGR2HLS: case CV_BGR2HSV_FULL: case CV_BGR2HLS_FULL:
  case CV_YUV2BGR_YV12: case CV_YUV2BGRA_YV12: case CV_YUV2BGR_IYUV: case CV_YUV2BGRA_IYUV:
  case CV_YUV2BGR_NV21: case CV_YUV2BGRA_NV21: case CV_YUV2BGR_NV12: case CV_YUV2BGRA_NV12:
  case CV_Lab2BGR: case CV_Luv2BGR: case CV_Lab2LBGR: case CV_Luv2LBGR:
  case CV_BGR2Lab: case CV_BGR2Luv: case CV_LBGR2Lab: case CV_LBGR2Luv:
  case CV_HSV2BGR: case CV_HLS2BGR: case CV_HSV2BGR_FULL: case CV_HLS2BGR_FULL:
  case CV_YUV2BGR_UYVY: case CV_YUV2BGRA_UYVY: case CV_YUV2BGR_YUY2:
  case CV_YUV2BGRA_YUY2:  case CV_YUV2BGR_YVYU: case CV_YUV2BGRA_YVYU:
  case CV_BGR2YUV_IYUV: case CV_BGRA2YUV_IYUV: case CV_BGR2YUV_YV12: case CV_BGRA2YUV_YV12:
      return false;
  default:
      return true;
  }
}

// clang-format on

// clang-format off
//////////////////////////////////////////////////////////////////////////////////////////
//                                   The main function                                  //
//////////////////////////////////////////////////////////////////////////////////////////
// L9694
void cviai::cvtColor(cv::InputArray _src, cv::OutputArray _dst, int code, int dcn) {
  int stype = _src.type();
  int scn = CV_MAT_CN(stype), depth = CV_MAT_DEPTH(stype);
  // int uidx, gbits, ycn; /* not used variable */

  cv::Mat src, dst;
  if (_src.getObj() == _dst.getObj())  // inplace processing (#6653)
    _src.copyTo(src);
  else
    src = _src.getMat();
  cv::Size sz = src.size();
  CV_Assert(depth == CV_8U || depth == CV_16U || depth == CV_32F);

  switch (code) {
    case CV_BGR2BGRA: case CV_RGB2BGRA: case CV_BGRA2BGR:
    case CV_RGBA2BGR: case CV_RGB2BGR: case CV_BGRA2RGBA:
      CV_Assert(scn == 3 || scn == 4);
      dcn = code == CV_BGR2BGRA || code == CV_RGB2BGRA || code == CV_BGRA2RGBA ? 4 : 3;
      _dst.create(sz, CV_MAKETYPE(depth, dcn));
      dst = _dst.getMat();
      cvtBGRtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows, depth, scn, dcn,
                  swapBlue(code));
      break;

    case CV_BGR2GRAY: case CV_BGRA2GRAY: case CV_RGB2GRAY: case CV_RGBA2GRAY:
      CV_Assert(scn == 3 || scn == 4);
      _dst.create(sz, CV_MAKETYPE(depth, 1));
      dst = _dst.getMat();
      cvtBGRtoGray(src.data, src.step, dst.data, dst.step, src.cols, src.rows, depth, scn,
                   swapBlue(code));
      break;

    case CV_GRAY2BGR: case CV_GRAY2BGRA:
      if (dcn <= 0) dcn = (code == CV_GRAY2BGRA) ? 4 : 3;
      CV_Assert(scn == 1 && (dcn == 3 || dcn == 4));
      _dst.create(sz, CV_MAKETYPE(depth, dcn));
      dst = _dst.getMat();
      cvtGraytoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows, depth, dcn);
      break;

    default:
      CV_Error(CV_StsBadFlag, "Unknown/unsupported color conversion code");
  }
}
// clang-format on