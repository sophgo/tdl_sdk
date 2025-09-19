#include "utils/image_alignment.hpp"

#include <cassert>
#include <iostream>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "utils/tdl_log.hpp"

#define EXT_FUNCTION 0
#undef SHRT_MIN
#define SHRT_MIN -32768
#undef SHRT_MAX
#define SHRT_MAX 32767
#undef USHRT_MAX
#define USHRT_MAX 65535
#undef UCHAR_MAX
#define UCHAR_MAX 255
typedef uint8_t uchar;

/* From imgproc.h */
//! interpolation algorithm
enum InterpolationFlags {
  /** nearest neighbor interpolation */
  INTER_NEAREST = 0,
  /** bilinear interpolation */
  INTER_LINEAR = 1,
  /** bicubic interpolation */
  INTER_CUBIC = 2,
  /** resampling using pixel area relation. It may be a preferred method for
  image decimation, as it gives moire'-free results. But when the image is
  zoomed, it is similar to the INTER_NEAREST method. */
  INTER_AREA = 3,
  /** Lanczos interpolation over 8x8 neighborhood */
  INTER_LANCZOS4 = 4,
  /** mask for interpolation codes */
  INTER_MAX = 7,
  /** flag, fills all of the destination image pixels. If some of them
  correspond to outliers in the source image, they are set to zero */
  WARP_FILL_OUTLIERS = 8,
  /** flag, inverse transformation

  For example, @ref cv::linearPolar or @ref cv::logPolar transforms:
  - flag is __not__ set: \f$dst( \rho , \phi ) = src(x,y)\f$
  - flag is set: \f$dst(x,y) = src( \rho , \phi )\f$
  */
  WARP_INVERSE_MAP = 16
};

/* From imgproc.h */
enum InterpolationMasks {
  INTER_BITS = 5,
  INTER_BITS2 = INTER_BITS * 2,
  INTER_TAB_SIZE = 1 << INTER_BITS,
  INTER_TAB_SIZE2 = INTER_TAB_SIZE * INTER_TAB_SIZE
};
enum BorderTypes {
  BORDER_CONSTANT = 0,   //!< `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
  BORDER_REPLICATE = 1,  //!< `aaaaaa|abcdefgh|hhhhhhh`
  BORDER_REFLECT = 2,    //!< `fedcba|abcdefgh|hgfedcb`
  BORDER_WRAP = 3,       //!< `cdefgh|abcdefgh|abcdefg`
  BORDER_REFLECT_101 = 4,  //!< `gfedcb|abcdefgh|gfedcba`
  BORDER_TRANSPARENT = 5,  //!< `uvwxyz|absdefgh|ijklmno`

  BORDER_REFLECT101 = BORDER_REFLECT_101,  //!< same as BORDER_REFLECT_101
  BORDER_DEFAULT = BORDER_REFLECT_101,     //!< same as BORDER_REFLECT_101
  BORDER_ISOLATED = 16                     //!< do not look outside of ROI
};
// #define DEBUG_IMGWARP

/************** interpolation formulas and tables ***************/
// L131
const int INTER_RESIZE_COEF_BITS = 11;
const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;

const int INTER_REMAP_COEF_BITS = 15;
const int INTER_REMAP_COEF_SCALE = 1 << INTER_REMAP_COEF_BITS;

static uchar NNDeltaTab_i[INTER_TAB_SIZE2][2];
static float BilinearTab_f[INTER_TAB_SIZE2][2][2];
static short BilinearTab_i[INTER_TAB_SIZE2][2][2];

short saturate_cast_short(int v) {
  return (short)((unsigned)(v - SHRT_MIN) <= (unsigned)USHRT_MAX
                     ? v
                     : v > 0 ? SHRT_MAX : SHRT_MIN);
}
uint8_t saturate_cast_uint8_t(int v) {
  return (uint8_t)((unsigned)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0);
}
uint8_t saturate_cast_uchar(int v) {
  return (uint8_t)((unsigned)v <= 255 ? v : v > 0 ? 255 : 0);
}

// L153
static inline void interpolateLinear(float x, float *coeffs) {
  coeffs[0] = 1.f - x;
  coeffs[1] = x;
}

int cv_borderInterpolate(int p, int len, int borderType) {
#ifdef CV_STATIC_ANALYSIS
  if (p >= 0 && p < len)
#else
  if ((unsigned)p < (unsigned)len)
#endif
    ;
  else if (borderType == BORDER_REPLICATE)
    p = p < 0 ? 0 : len - 1;
  else if (borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101) {
    int delta = borderType == BORDER_REFLECT_101;
    if (len == 1) return 0;
    do {
      if (p < 0)
        p = -p - 1 + delta;
      else
        p = len - 1 - (p - len) - delta;
    }
#ifdef CV_STATIC_ANALYSIS
    while (p < 0 || p >= len);
#else
    while ((unsigned)p >= (unsigned)len);
#endif
  } else if (borderType == BORDER_WRAP) {
    assert(len > 0);
    if (p < 0) p -= ((p - len + 1) / len) * len;
    if (p >= len) p %= len;
  } else if (borderType == BORDER_CONSTANT)
    p = -1;
  else
    assert(0);  // CV_StsBadArg, "Unknown/unsupported border type" );
  return p;
}
// L197
static void initInterTab1D(int method, float *tab, int tabsz) {
  float scale = 1.f / tabsz;
  if (method == INTER_LINEAR) {
    for (int i = 0; i < tabsz; i++, tab += 2) interpolateLinear(i * scale, tab);
  } else if (method == INTER_CUBIC) {
    assert(0);
  } else if (method == INTER_LANCZOS4) {
    assert(0);
  } else
    assert(0);  // CV_Error(CV_StsBadArg, "Unknown interpolation method");
}

// L220
static const void *initInterTab2D(int method, bool fixpt) {
  static bool inittab[INTER_MAX + 1] = {false};
  float *tab = 0;
  short *itab = 0;
  int ksize = 0;
  if (method == INTER_LINEAR) {
    tab = BilinearTab_f[0][0], itab = BilinearTab_i[0][0], ksize = 2;
  } else if (method == INTER_CUBIC) {
    assert(0);
  } else if (method == INTER_LANCZOS4) {
    assert(0);
  } else
    assert(0);  // CV_Error(CV_StsBadArg, "Unknown/unsupported interpolation
                // type");

  if (!inittab[method]) {
    float *_tab = new float[8 * INTER_TAB_SIZE];
    // cv::AutoBuffer<float> _tab(8 * INTER_TAB_SIZE);
    int i, j, k1, k2;
    initInterTab1D(method, _tab, INTER_TAB_SIZE);
    for (i = 0; i < INTER_TAB_SIZE; i++)
      for (j = 0; j < INTER_TAB_SIZE;
           j++, tab += ksize * ksize, itab += ksize * ksize) {
        int isum = 0;
        NNDeltaTab_i[i * INTER_TAB_SIZE + j][0] = j < INTER_TAB_SIZE / 2;
        NNDeltaTab_i[i * INTER_TAB_SIZE + j][1] = i < INTER_TAB_SIZE / 2;

        for (k1 = 0; k1 < ksize; k1++) {
          float vy = _tab[i * ksize + k1];
          for (k2 = 0; k2 < ksize; k2++) {
            float v = vy * _tab[j * ksize + k2];
            tab[k1 * ksize + k2] = v;
            isum += itab[k1 * ksize + k2] =
                saturate_cast_short(v * INTER_REMAP_COEF_SCALE);
          }
        }

        if (isum != INTER_REMAP_COEF_SCALE) {
          int diff = isum - INTER_REMAP_COEF_SCALE;
          int ksize2 = ksize / 2, Mk1 = ksize2, Mk2 = ksize2, mk1 = ksize2,
              mk2 = ksize2;
          for (k1 = ksize2; k1 < ksize2 + 2; k1++)
            for (k2 = ksize2; k2 < ksize2 + 2; k2++) {
              if (itab[k1 * ksize + k2] < itab[mk1 * ksize + mk2])
                mk1 = k1, mk2 = k2;
              else if (itab[k1 * ksize + k2] > itab[Mk1 * ksize + Mk2])
                Mk1 = k1, Mk2 = k2;
            }
          if (diff < 0)
            itab[Mk1 * ksize + Mk2] = (short)(itab[Mk1 * ksize + Mk2] - diff);
          else
            itab[mk1 * ksize + mk2] = (short)(itab[mk1 * ksize + mk2] - diff);
        }
      }
    tab -= INTER_TAB_SIZE2 * ksize * ksize;
    itab -= INTER_TAB_SIZE2 * ksize * ksize;
#if CV_NEON
    if (method == INTER_LINEAR) {
      for (i = 0; i < INTER_TAB_SIZE2; i++)
        for (j = 0; j < 4; j++) {
          BilinearTab_iC4[i][0][j * 2] = BilinearTab_i[i][0][0];
          BilinearTab_iC4[i][0][j * 2 + 1] = BilinearTab_i[i][0][1];
          BilinearTab_iC4[i][1][j * 2] = BilinearTab_i[i][1][0];
          BilinearTab_iC4[i][1][j * 2 + 1] = BilinearTab_i[i][1][1];
        }
    }
#endif
    inittab[method] = true;
    delete _tab;
  }
  return fixpt ? (const void *)itab : (const void *)tab;
}

template <typename ST, typename DT, int bits>
struct FixedPtCast {
  typedef ST type1;
  typedef DT rtype;
  enum { SHIFT = bits, DELTA = 1 << (bits - 1) };

  DT operator()(ST val) const {
    // return cv::saturate_cast<DT>((val + DELTA) >> SHIFT);
    return saturate_cast_uchar((val + DELTA) >> SHIFT);
  }
};

template <class CastOp, typename AT>
static void remapBilinear_impl(const uchar *S0, int src_width, int src_height,
                               int src_stride, uchar *p_dst_img, int dst_width,
                               int dst_height, int dst_stride,
                               const short *p_xy, int xy_width, int xy_height,
                               int xy_stride, const uint16_t *p_fxy,
                               int fxy_width, int fxy_height, int fxy_stride,
                               const void *_wtab, int borderType,
                               const uchar *borderValue) {
  typedef typename CastOp::rtype T;   // uchar
  typedef typename CastOp::type1 WT;  // int
  int k, cn = 3;                      //_src.channels();
  const AT *wtab = (const AT *)_wtab;
  // const T* S0 = _src.ptr<T>();
  // size_t sstep = _src.step / sizeof(S0[0]);
  int sstep = src_stride;
  T cval[3];
  int dx, dy;
  CastOp castOp;
  // VecOp vecOp;
  // std::cout<<"to remap bilinear,_fxy
  // channels:"<<_fxy.channels()<<",type:"<<_fxy.type()<<std::endl;
  for (k = 0; k < cn; k++) cval[k] = saturate_cast_uchar(borderValue[k & 3]);
  int ssize_width = src_width;
  int ssize_height = src_height;

  // int dsize_width = dst_width;
  // int dsize_height = dst_height;

  unsigned width1 = std::max(src_width - 1, 0),
           height1 = std::max(src_height - 1, 0);
  // assert(ssize.area() > 0);

  for (dy = 0; dy < dst_height; dy++) {
    uchar *D = p_dst_img + dy * dst_stride;
    const short *XY = p_xy + dy * xy_stride;        //_xy.ptr<short>(dy);
    const uint16_t *FXY = p_fxy + dy * fxy_stride;  //_fxy.ptr<uint16_t>(dy);
    int X0 = 0;
    bool prevInlier = false;

    for (dx = 0; dx <= dst_width; dx++) {
      bool curInlier = dx < dst_width ? (unsigned)XY[dx * 2] < width1 &&
                                            (unsigned)XY[dx * 2 + 1] < height1
                                      : !prevInlier;
      if (curInlier == prevInlier) continue;

      int X1 = dx;
      dx = X0;
      X0 = X1;
      prevInlier = curInlier;

      if (!curInlier) {
        int len = 0;  // vecOp(_src, D, XY + dx * 2, FXY + dx, wtab, X1 - dx);
        D += len * cn;
        dx += len;

        if (cn == 1) {
          assert(0);
        } else if (cn == 2) {
          assert(0);
        } else if (cn == 3)
          for (; dx < X1; dx++, D += 3) {
            int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
            const AT *w = wtab + FXY[dx] * 4;
            const T *S = S0 + sy * sstep + sx * 3;
            WT t0 = S[0] * w[0] + S[3] * w[1] + S[sstep] * w[2] +
                    S[sstep + 3] * w[3];
            WT t1 = S[1] * w[0] + S[4] * w[1] + S[sstep + 1] * w[2] +
                    S[sstep + 4] * w[3];
            WT t2 = S[2] * w[0] + S[5] * w[1] + S[sstep + 2] * w[2] +
                    S[sstep + 5] * w[3];
            D[0] = castOp(t0);
            D[1] = castOp(t1);
            D[2] = castOp(t2);
          }
        else if (cn == 4) {
          assert(0);
        } else {
          assert(0);
        }
      } else {
        if (borderType == BORDER_TRANSPARENT && cn != 3) {
          assert(0);
        }

        if (cn == 1) {
          assert(0);
        } else
          for (; dx < X1; dx++, D += cn) {
            int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
            if (borderType == BORDER_CONSTANT &&
                (sx >= ssize_width || sx + 1 < 0 || sy >= ssize_height ||
                 sy + 1 < 0)) {
              for (k = 0; k < cn; k++) D[k] = cval[k];
            } else {
              int sx0, sx1, sy0, sy1;
              const T *v0, *v1, *v2, *v3;
              const AT *w = wtab + FXY[dx] * 4;
#if 0
              if (borderType == cv::BORDER_REPLICATE) {
                assert(0);
              } else if (borderType == cv::BORDER_TRANSPARENT &&
                         ((unsigned)sx >= (unsigned)(ssize_width - 1) ||
                          (unsigned)sy >= (unsigned)(ssize_height - 1))) {
                assert(0);
              } else {
                /* this case*/
              }
#endif

#if 1 /* this case */
              sx0 = cv_borderInterpolate(sx, ssize_width, borderType);
              sx1 = cv_borderInterpolate(sx + 1, ssize_width, borderType);
              sy0 = cv_borderInterpolate(sy, ssize_height, borderType);
              sy1 = cv_borderInterpolate(sy + 1, ssize_height, borderType);
              v0 =
                  sx0 >= 0 && sy0 >= 0 ? S0 + sy0 * sstep + sx0 * cn : &cval[0];
              v1 =
                  sx1 >= 0 && sy0 >= 0 ? S0 + sy0 * sstep + sx1 * cn : &cval[0];
              v2 =
                  sx0 >= 0 && sy1 >= 0 ? S0 + sy1 * sstep + sx0 * cn : &cval[0];
              v3 =
                  sx1 >= 0 && sy1 >= 0 ? S0 + sy1 * sstep + sx1 * cn : &cval[0];
#endif
              for (k = 0; k < cn; k++)
                D[k] = castOp(WT(v0[k] * w[0] + v1[k] * w[1] + v2[k] * w[2] +
                                 v3[k] * w[3]));
            }
          }
      }
    }
  }
}

// void remap_bilinear(const cv::Mat* src, cv::Mat* dst, const cv::Mat* m1,
// const cv::Mat* m2,
//                int borderType, const cv::Scalar& borderValue,
//                const cv::Range& range){
void remap_bilinear(const uchar *p_src_img, int src_width, int src_height,
                    int src_stride, int src_channel, uchar *p_dst,
                    int dst_width, int dst_height, int dst_stride,
                    int dst_channel, short *p_m1, int m1_width, int m1_height,
                    int m1_stride, int m1_channel, uint16_t *p_m2, int m2_width,
                    int m2_height, int m2_stride, int m2_channel,
                    int borderType, const uchar *borderValue, int range_start,
                    int range_end) {
  // std::cout << "range start:" << range_start << ",end:" << range_end <<
  // std::endl;
  int x, y, x1, y1;
  const int buf_size = 1 << 14;
  int brows0 = std::min(128, dst_width);  //, map_depth = m1->depth();
  int bcols0 = std::min(buf_size / brows0, dst_width);
  brows0 = std::min(buf_size / bcols0, dst_height);

  // cv::Mat _bufxy(brows0, bcols0, CV_16SC2);
  // cv::Mat  _bufa;
  // _bufa.create(brows0, bcols0, CV_16UC1);
  uint16_t *pbufa = new uint16_t[brows0 * bcols0];
  bool planar_input = false;
  const void *ctab = initInterTab2D(INTER_LINEAR, true);
  for (y = range_start; y < range_end; y += brows0) {
    for (x = 0; x < dst_width; x += bcols0) {
      int brows = std::min(brows0, range_end - y);
      int bcols = std::min(bcols0, dst_width - x);
      uchar *p_dst_roi = p_dst + y * dst_stride + x * dst_channel;  //
      uint16_t *pbufa_roi = pbufa;
      // update bufa in lower loop
      short *pbuf_xy_roi = nullptr;
      for (y1 = 0; y1 < brows; y1++) {
        // short* XY = bufxy.ptr<short>(y1);        /* not used */
        uint16_t *A = pbufa_roi + y1 * bcols0;  // bufa.ptr<uint16_t>(y1);

        if (true) {
          // bufxy = (*m1)(cv::Rect(x, y, bcols, brows));
          pbuf_xy_roi = p_m1 + y * m1_stride + x * m1_channel;
          // const uint16_t* sA = m2->ptr<uint16_t>(y + y1) + x;
          const uint16_t *sA = p_m2 + (y + y1) * m2_stride + x;
          x1 = 0;

#if CV_NEON
          uint16x8_t v_scale = vdupq_n_u16(INTER_TAB_SIZE2 - 1);
          for (; x1 <= bcols - 8; x1 += 8)
            vst1q_u16(A + x1, vandq_u16(vld1q_u16(sA + x1), v_scale));
#endif

          for (; x1 < bcols; x1++)
            A[x1] = (uint16_t)(sA[x1] & (INTER_TAB_SIZE2 - 1));
        } else if (planar_input) {
          assert(0);
        } else {
          assert(0);
        }
      }
      remapBilinear_impl<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, short>(
          p_src_img, src_width, src_height, src_stride, p_dst_roi, bcols, brows,
          dst_stride, pbuf_xy_roi, bcols, brows, m1_stride, pbufa_roi, bcols,
          brows, bcols0, ctab, borderType, borderValue);
      // bufxy, bufa, ctab, borderType, borderValue);
    }
  }
}

// L5507
int cv_round_src(double value) {
  return (int)(value + (value >= 0 ? 0.5 : -0.5));
}
void WarpAffineInvoker_impl(const uchar *p_src_img, int src_width,
                            int src_height, int src_stride, int src_channel,
                            uchar *p_dst, int dst_width, int dst_height,
                            int dst_stride, int dst_channel, int interpolation,
                            int borderType, uchar *border_val, int *adelta,
                            int *bdelta, const double *M) {
  const int BLOCK_SZ = 64;
  short XY[BLOCK_SZ * BLOCK_SZ * 2], A[BLOCK_SZ * BLOCK_SZ];
  const int AB_BITS = std::max(10, (int)INTER_BITS);
  const int AB_SCALE = 1 << AB_BITS;
  int round_delta = interpolation == INTER_NEAREST
                        ? AB_SCALE / 2
                        : AB_SCALE / INTER_TAB_SIZE / 2,
      x, y, x1, y1;

  int bh0 = std::min(BLOCK_SZ / 2, dst_width);
  int bw0 = std::min(BLOCK_SZ * BLOCK_SZ / bh0, dst_width);
  bh0 = std::min(BLOCK_SZ * BLOCK_SZ / bw0, dst_height);
  int range_start = 0;
  int range_end = dst_height;

  // uchar border_val[4];
  // for (int k = 0; k < 3; k++) border_val[k] =
  // saturate_cast_uchar(borderValue[k & 3]);
  for (y = range_start; y < range_end; y += bh0) {
    for (x = 0; x < dst_width; x += bw0) {
      int bw = std::min(bw0, dst_width - x);
      int bh = std::min(bh0, range_end - y);

      // cv::Mat _XY(bh, bw, CV_16SC2, XY), matA;
      // cv::Mat dpart(dst, cv::Rect(x, y, bw, bh));
      uchar *p_dpart = p_dst + y * dst_stride + x * dst_channel;
      for (y1 = 0; y1 < bh; y1++) {
        short *xy = XY + y1 * bw * 2;
        int X0 =
            cv_round_src((M[1] * (y + y1) + M[2]) * AB_SCALE) + round_delta;
        int Y0 =
            cv_round_src((M[4] * (y + y1) + M[5]) * AB_SCALE) + round_delta;

        if (interpolation == INTER_NEAREST) {
          x1 = 0;
#if CV_NEON
          int32x4_t v_X0 = vdupq_n_s32(X0), v_Y0 = vdupq_n_s32(Y0);
          for (; x1 <= bw - 8; x1 += 8) {
            int16x8x2_t v_dst;
            v_dst.val[0] = vcombine_s16(
                vqmovn_s32(vshrq_n_s32(
                    vaddq_s32(v_X0, vld1q_s32(adelta + x + x1)), AB_BITS)),
                vqmovn_s32(vshrq_n_s32(
                    vaddq_s32(v_X0, vld1q_s32(adelta + x + x1 + 4)), AB_BITS)));
            v_dst.val[1] = vcombine_s16(
                vqmovn_s32(vshrq_n_s32(
                    vaddq_s32(v_Y0, vld1q_s32(bdelta + x + x1)), AB_BITS)),
                vqmovn_s32(vshrq_n_s32(
                    vaddq_s32(v_Y0, vld1q_s32(bdelta + x + x1 + 4)), AB_BITS)));

            vst2q_s16(xy + (x1 << 1), v_dst);
          }
#endif
          for (; x1 < bw; x1++) {
            int X = (X0 + adelta[x + x1]) >> AB_BITS;
            int Y = (Y0 + bdelta[x + x1]) >> AB_BITS;
            xy[x1 * 2] = saturate_cast_short(X);
            xy[x1 * 2 + 1] = saturate_cast_short(Y);
          }
        } else {
          short *alpha = A + y1 * bw;
          x1 = 0;
#if CV_NEON
          int32x4_t v__X0 = vdupq_n_s32(X0), v__Y0 = vdupq_n_s32(Y0),
                    v_mask = vdupq_n_s32(INTER_TAB_SIZE - 1);
          for (; x1 <= bw - 8; x1 += 8) {
            int32x4_t v_X0 =
                vshrq_n_s32(vaddq_s32(v__X0, vld1q_s32(adelta + x + x1)),
                            AB_BITS - INTER_BITS);
            int32x4_t v_Y0 =
                vshrq_n_s32(vaddq_s32(v__Y0, vld1q_s32(bdelta + x + x1)),
                            AB_BITS - INTER_BITS);
            int32x4_t v_X1 =
                vshrq_n_s32(vaddq_s32(v__X0, vld1q_s32(adelta + x + x1 + 4)),
                            AB_BITS - INTER_BITS);
            int32x4_t v_Y1 =
                vshrq_n_s32(vaddq_s32(v__Y0, vld1q_s32(bdelta + x + x1 + 4)),
                            AB_BITS - INTER_BITS);

            int16x8x2_t v_xy;
            v_xy.val[0] =
                vcombine_s16(vqmovn_s32(vshrq_n_s32(v_X0, INTER_BITS)),
                             vqmovn_s32(vshrq_n_s32(v_X1, INTER_BITS)));
            v_xy.val[1] =
                vcombine_s16(vqmovn_s32(vshrq_n_s32(v_Y0, INTER_BITS)),
                             vqmovn_s32(vshrq_n_s32(v_Y1, INTER_BITS)));

            vst2q_s16(xy + (x1 << 1), v_xy);

            int16x4_t v_alpha0 = vmovn_s32(
                vaddq_s32(vshlq_n_s32(vandq_s32(v_Y0, v_mask), INTER_BITS),
                          vandq_s32(v_X0, v_mask)));
            int16x4_t v_alpha1 = vmovn_s32(
                vaddq_s32(vshlq_n_s32(vandq_s32(v_Y1, v_mask), INTER_BITS),
                          vandq_s32(v_X1, v_mask)));
            vst1q_s16(alpha + x1, vcombine_s16(v_alpha0, v_alpha1));
          }
#endif
          for (; x1 < bw; x1++) {
            int X = (X0 + adelta[x + x1]) >> (AB_BITS - INTER_BITS);
            int Y = (Y0 + bdelta[x + x1]) >> (AB_BITS - INTER_BITS);
            xy[x1 * 2] = saturate_cast_short(X >> INTER_BITS);
            xy[x1 * 2 + 1] = saturate_cast_short(Y >> INTER_BITS);
            alpha[x1] = (short)((Y & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE +
                                (X & (INTER_TAB_SIZE - 1)));
          }
        }
      }

      if (interpolation == INTER_NEAREST) {
      }
      // remap(src, dpart, _XY, cv::Mat(), interpolation, borderType,
      // borderValue);
      else {
        // cv::Mat _matA(bh, bw, CV_16U, A);
        // uint16_t *pmata = new uint16_t[bh*bw];
        // std::cout << "intertype:" << interpolation << std::endl;
        // remap(src, dpart, _XY, _matA, interpolation, borderType,
        // borderValue);
        // remap_bilinear(&src,&dpart,&_XY,&_matA,borderType,borderValue,cv::Range(0,
        // dpart.rows));
        remap_bilinear(p_src_img, src_width, src_height, src_stride,
                       src_channel, p_dpart, bw, bh, dst_stride, dst_channel,
                       XY, bw, bh, bw * 2, 2, (uint16_t *)A, bw, bh, bw, 1,
                       borderType, border_val, 0, bh);
      }
    }
  }
}

// L5926
void _warpAffine(const uchar *src_data, size_t src_step, int src_width,
                 int src_height, uchar *dst_data, size_t dst_step,
                 int dst_width, int dst_height, const double M[6],
                 int interpolation, int borderType,
                 const double borderValue[4]) {
  int x;
  // cv::AutoBuffer<int> _abdelta(dst_width * 2);
  int *_abdelta = new int[dst_width * 2];
  int *adelta = &_abdelta[0], *bdelta = adelta + dst_width;
  const int AB_BITS = std::max(10, (int)INTER_BITS);
  const int AB_SCALE = 1 << AB_BITS;

  for (x = 0; x < dst_width; x++) {
    adelta[x] = cv_round_src(M[0] * x * AB_SCALE);
    bdelta[x] = cv_round_src(M[3] * x * AB_SCALE);
  }
  uchar border_val[4] = {0};
  int src_channels = 3;
  int dst_channels = 3;
  WarpAffineInvoker_impl(src_data, src_width, src_height, src_step,
                         src_channels, dst_data, dst_width, dst_height,
                         dst_step, dst_channels, interpolation, borderType,
                         border_val, adelta, bdelta, M);
  delete[] _abdelta;
}

// L5959
void tdl_warp_affine(const unsigned char *src_data,
                     const float *affine_transform, unsigned int src_step,
                     int src_width, int src_height, unsigned char *dst_data,
                     unsigned int dst_step, int dst_width, int dst_height) {
  double M[6];
  // cv::Mat matM(2, 3, CV_64F, M);
  int flags = INTER_LINEAR;
  int interpolation = INTER_LINEAR;
  int borderType = BORDER_CONSTANT;
  double borderValue[4] = {0};
  if (interpolation == INTER_AREA) interpolation = INTER_LINEAR;

  // assert((M0.type() == CV_32F || M0.type() == CV_64F) && M0.rows == 2 &&
  // M0.cols == 3); M0.convertTo(matM, matM.type());
  for (int i = 0; i < 6; i++) {
    M[i] = affine_transform[i];
  }
  if (!(flags & WARP_INVERSE_MAP)) {
    double D = M[0] * M[4] - M[1] * M[3];
    D = D != 0 ? 1. / D : 0;
    double A11 = M[4] * D, A22 = M[0] * D;
    M[0] = A11;
    M[1] *= -D;
    M[3] *= -D;
    M[4] = A22;
    double b1 = -M[0] * M[2] - M[1] * M[5];
    double b2 = -M[3] * M[2] - M[4] * M[5];
    M[2] = b1;
    M[5] = b2;
  }

  // printf("[warpAffine] flags = %d, borderType = %d\n", flags, borderType);
  // std::cout << "dststep:" << dst_step << ",srcstep:" << src_step <<
  // std::endl;
  _warpAffine(src_data, src_step, src_width, src_height, dst_data, dst_step,
              dst_width, dst_height, M, interpolation, borderType, borderValue);
}

/**
 * Solve similarity transform matrix (no reflection)
 */
Eigen::Matrix<float, 4, 1> solve_stm_no_reflection(const float *src_pts,
                                                   const float *dst_pts,
                                                   int num_points,
                                                   float *distance) {
  Eigen::MatrixXf A(2 * num_points, 4);
  Eigen::VectorXf b(2 * num_points);

  for (int i = 0; i < num_points; ++i) {
    float sx = src_pts[2 * i];
    float sy = src_pts[2 * i + 1];
    float dx = dst_pts[2 * i];
    float dy = dst_pts[2 * i + 1];

    // 上半部分
    A(i, 0) = sx;
    A(i, 1) = sy;
    A(i, 2) = 1.0f;
    A(i, 3) = 0.0f;
    b(i) = dx;

    // 下半部分
    A(i + num_points, 0) = sy;
    A(i + num_points, 1) = -sx;
    A(i + num_points, 2) = 0.0f;
    A(i + num_points, 3) = 1.0f;
    b(i + num_points) = dy;
  }

  // 2. 求解最小二乘问题 x = (A^T A)^-1 A^T b
  //    也可以用 QR/SVD 等分解
  Eigen::Vector4f x = (A.transpose() * A).ldlt().solve(A.transpose() * b);

  // 3. 可选：计算残差 (A*x - b).squaredNorm()
  if (distance != nullptr) {
    *distance = (A * x - b).squaredNorm();
  }

  return x;  // [cosθ, sinθ, tx, ty]
}

int tdl_get_similarity_transform_matrix(const float *src_pts_xy,
                                        const float *dst_pts_xy, int num_points,
                                        float *transform) {
  // if (num_points < 5) {
  //   LOGE("get_similarity_transform_matrix num_points must be greater than
  //   5"); return -1;
  // }
  float d_0;
  // [cosθ, sinθ, tx, ty]
  Eigen::Matrix<float, 4, 1> stm =
      solve_stm_no_reflection(src_pts_xy, dst_pts_xy, num_points, &d_0);

  transform[0] = stm[0];
  transform[1] = stm[1];
  transform[2] = stm[2];
  transform[3] = -stm[1];
  transform[4] = stm[0];
  transform[5] = stm[3];
  return 0;
}

int get_face_transform(const float *landmark_pts, const int width,
                       float *transform) {
  assert(width == 96 || width == 112 || width == 64);
  // assert(height == 112 || (width == 64 && height == 64));

  // int ref_width = width;

  std::vector<float> refer_pts;
  if (96 == width) {
    refer_pts = {30.29459953, 51.69630051, 65.53179932, 51.50139999,
                 48.02519989, 71.73660278, 33.54930115, 92.3655014,
                 62.72990036, 92.20410156};
  } else if (112 == width) {
    // refer_pts = {38.4066, 51.5989, 73.6438, 51.5989, 56.0252,
    //              71.7366, 41.4349, 92.2848, 70.6155, 92.2848};
    refer_pts = {38.29459953, 51.69630051, 73.53179932, 51.50139999,
                 56.02519989, 71.73660278, 41.54930115, 92.3655014,
                 70.72990036, 92.20410156};
  } else if (64 == width) {
    refer_pts = {21.88262830, 29.53926611, 42.01607013, 29.42789995,
                 32.01279921, 40.99029482, 23.74127067, 45.7776475,
                 40.41506506, 45.68542363};
  } else {
    LOGE("unsupported face image width: %d", width);
    return -1;
  }

  int ret = tdl_get_similarity_transform_matrix(landmark_pts, &refer_pts[0], 5,
                                                transform);
  return ret;
}

int get_license_plate_transform(const float *landmark_pts, float *transform) {
  std::vector<float> refer_pts;

  refer_pts = {0.0, 0.0, 96.0, 0.0, 96.0, 24.0, 0.0, 24.0};

  int ret = tdl_get_similarity_transform_matrix(landmark_pts, &refer_pts[0], 4,
                                                transform);
  return ret;
}

#ifndef NO_OPEN_CV
#include <opencv2/opencv.hpp>

cv::Mat tformfwd(const cv::Mat &trans, const cv::Mat &uv) {
  cv::Mat uv_h = cv::Mat::ones(uv.rows, 3, CV_64FC1);
  uv.copyTo(uv_h(cv::Rect(0, 0, 2, uv.rows)));
  cv::Mat xv_h = uv_h * trans.t();
  return xv_h(cv::Rect(0, 0, 2, uv.rows));
}

cv::Mat find_none_flectives_similarity(const cv::Mat &xy, const cv::Mat &uv) {
  cv::Mat A = cv::Mat::zeros(2 * xy.rows, 4, CV_64FC1);
  cv::Mat b = cv::Mat::zeros(2 * xy.rows, 1, CV_64FC1);
  cv::Mat x = cv::Mat::zeros(4, 1, CV_64FC1);

  xy(cv::Rect(0, 0, 1, xy.rows)).copyTo(A(cv::Rect(0, 0, 1, xy.rows)));  // x
  xy(cv::Rect(1, 0, 1, xy.rows)).copyTo(A(cv::Rect(1, 0, 1, xy.rows)));  // y
  A(cv::Rect(2, 0, 1, xy.rows)).setTo(1.);

  xy(cv::Rect(1, 0, 1, xy.rows))
      .copyTo(A(cv::Rect(0, xy.rows, 1, xy.rows)));  // y
  xy(cv::Rect(0, 0, 1, xy.rows))
      .copyTo(A(cv::Rect(1, xy.rows, 1, xy.rows)));  //-x
  A(cv::Rect(1, xy.rows, 1, xy.rows)) *= -1;
  A(cv::Rect(3, xy.rows, 1, xy.rows)).setTo(1.);

  uv(cv::Rect(0, 0, 1, uv.rows)).copyTo(b(cv::Rect(0, 0, 1, uv.rows)));
  uv(cv::Rect(1, 0, 1, uv.rows)).copyTo(b(cv::Rect(0, uv.rows, 1, uv.rows)));

  cv::solve(A, b, x, cv::DECOMP_SVD);
  cv::Mat trans = (cv::Mat_<double>(3, 3) << x.at<double>(0), x.at<double>(1),
                   x.at<double>(2), -x.at<double>(1), x.at<double>(0),
                   x.at<double>(3), 0, 0, 1);

  return trans;
}

cv::Mat find_similarity(const cv::Mat &xy, const cv::Mat &uv) {
  cv::Mat trans1 = find_none_flectives_similarity(xy, uv);
  cv::Mat xy_reflect = xy.clone();
  xy_reflect(cv::Rect(0, 0, 1, xy.rows)) *= -1;
  cv::Mat trans2r = find_none_flectives_similarity(xy_reflect, uv);
  cv::Mat reflect = (cv::Mat_<double>(3, 3) << -1, 0, 0, 0, 1, 0, 0, 0, 1);

  cv::Mat trans2 = trans2r * reflect;
  cv::Mat xy1 = tformfwd(trans1, xy);
  double norm1 = cv::norm(xy1 - uv);

  cv::Mat xy2 = tformfwd(trans2, xy);
  double norm2 = cv::norm(xy2 - uv);

  cv::Mat trans;
  if (norm1 < norm2) {
    trans = trans1;
  } else {
    trans = trans2;
  }
  return trans;
}

cv::Mat get_similarity_transform_opencv(
    const std::vector<cv::Point2f> &src_points,
    const std::vector<cv::Point2f> &dst_points, bool reflective = true) {
  cv::Mat trans;
  cv::Mat src((int)src_points.size(), 2, CV_32FC1, (void *)(&src_points[0].x));
  src.convertTo(src, CV_64FC1);

  cv::Mat dst((int)dst_points.size(), 2, CV_32FC1, (void *)(&dst_points[0].x));
  dst.convertTo(dst, CV_64FC1);

  if (reflective) {
    trans = find_similarity(src, dst);
  } else {
    trans = find_none_flectives_similarity(src, dst);
  }
  cv::Mat trans_cv2 = trans(cv::Rect(0, 0, trans.rows, 2));
  return trans_cv2;
}

int32_t tdl_warp_affine_opencv(const unsigned char *src_data,
                               const float *affine_transform,
                               unsigned int src_step, int src_width,
                               int src_height, unsigned char *dst_data,
                               unsigned int dst_step, int dst_width,
                               int dst_height) {
  cv::Mat src(src_height, src_width, CV_8UC3,
              const_cast<unsigned char *>(src_data), src_step);
  cv::Mat dst(dst_height, dst_width, CV_8UC3, dst_data, dst_step);
  cv::Mat trans = cv::Mat::zeros(2, 3, CV_64F);
  for (int i = 0; i < 6; i++) {
    trans.at<double>(i) = affine_transform[i];
  }
  cv::warpAffine(src, dst, trans, dst.size(), cv::INTER_NEAREST);
  return 0;
}
#endif

int32_t tdl_face_warp_affine(const unsigned char *src_data,
                             unsigned int src_step, int src_width,
                             int src_height, unsigned char *dst_data,
                             unsigned int dst_step, int dst_width,
                             int dst_height, const float *src_pts5_xy) {
  cv::Mat src(src_height, src_width, CV_8UC3,
              const_cast<unsigned char *>(src_data), src_step);
  cv::Mat dst(dst_height, dst_width, CV_8UC3, dst_data, dst_step);

  float transform[6];
  int ret = get_face_transform(src_pts5_xy, 112, transform);
  if (ret != 0) {
    LOGE("get_face_transform failed");
    return ret;
  }
  cv::Mat trans = cv::Mat::zeros(2, 3, CV_64F);
  for (int i = 0; i < 6; i++) {
    trans.at<double>(i) = transform[i];
  }
  cv::warpAffine(src, dst, trans, dst.size(), cv::INTER_LINEAR);

  // tdl_warp_affine(src_data, transform, src_step, src_width, src_height,
  //                 dst_data, dst_step, dst_width, dst_height);

  return 0;
}

int32_t tdl_license_plate_warp_affine(const unsigned char *src_data,
                                      unsigned int src_step, int src_width,
                                      int src_height, unsigned char *dst_data,
                                      unsigned int dst_step, int dst_width,
                                      int dst_height,
                                      const float *src_pts4_xy) {
  cv::Mat src(src_height, src_width, CV_8UC3,
              const_cast<unsigned char *>(src_data), src_step);
  cv::Mat dst(dst_height, dst_width, CV_8UC3, dst_data, dst_step);

  float transform[6];
  int ret = get_license_plate_transform(src_pts4_xy, transform);
  if (ret != 0) {
    LOGE("get_face_transform failed");
    return ret;
  }
  cv::Mat trans = cv::Mat::zeros(2, 3, CV_64F);
  for (int i = 0; i < 6; i++) {
    trans.at<double>(i) = transform[i];
  }
  cv::warpAffine(src, dst, trans, dst.size(), cv::INTER_LINEAR);

  return 0;
}