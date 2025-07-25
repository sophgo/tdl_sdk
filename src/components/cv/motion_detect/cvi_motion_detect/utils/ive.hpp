#pragma once

#include <limits>
#include <memory>
#include <vector>

#include "cvi_comm.hpp"
#include "ive_common.hpp"

namespace ive {

class IVEImageImpl;
class IVEImpl;
class IVE;

class IVEImage {
 public:
  IVEImage();
  ~IVEImage();
  IVEImage(const IVEImage &other) = delete;
  IVEImage &operator=(const IVEImage &other) = delete;

  IVEImageImpl *getImpl();

  int32_t bufFlush(IVE *ive_instance);
  int32_t bufRequest(IVE *ive_instance);
  int32_t create(IVE *ive_instance, ImageType enType, CVI_U16 u16Width,
                 CVI_U16 u16Height, bool cached = false);
  int32_t create(IVE *ive_instance, ImageType enType, CVI_U16 u16Width,
                 CVI_U16 u16Height, IVEImage *buf, bool cached = false);
  int32_t create(IVE *ive_instance);
  int32_t free();
  // int32_t toFrame(VIDEO_FRAME_INFO_S *frame, bool invertPackage = false);
  int32_t fromFrame(VIDEO_FRAME_INFO_S *frame);
  int32_t write(const std::string &fname);
  CVI_U32 getHeight();
  CVI_U32 getWidth();
  int32_t setZero(IVE *ive_instance);
  std::vector<CVI_U32> getStride();
  std::vector<CVI_U8 *> getVAddr();
  std::vector<CVI_U64> getPAddr();
  ImageType getType();

 private:
  std::shared_ptr<IVEImageImpl> mp_impl_;
};

class IVE {
 public:
  IVE();
  ~IVE();
  IVE(const IVE &other) = delete;
  IVE &operator=(const IVE &other) = delete;

  int32_t init();
  int32_t destroy();
  CVI_U32 getAlignedWidth(uint32_t width);
  int32_t dma(IVEImage *pSrc, IVEImage *pDst, DMAMode mode = DIRECT_COPY,
              CVI_U64 u64Val = 0, CVI_U8 u8HorSegSize = 0,
              CVI_U8 u8ElemSize = 0, CVI_U8 u8VerSegRows = 0);
  // int32_t sub(IVEImage *pSrc1, IVEImage *pSrc2, IVEImage *pDst,
  //             SubMode mode = ABS);
  // int32_t roi(IVEImage *pSrc, IVEImage *pDst, uint32_t x1, uint32_t x2,
  //             uint32_t y1, uint32_t y2);
  // int32_t andImage(IVEImage *pSrc1, IVEImage *pSrc2, IVEImage *pDst);
  // int32_t orImage(IVEImage *pSrc1, IVEImage *pSrc2, IVEImage *pDst);
  // int32_t erode(IVEImage *pSrc1, IVEImage *pDst,
  //               const std::vector<int32_t> &mask);
  // int32_t dilate(IVEImage *pSrc1, IVEImage *pDst,
  //                const std::vector<int32_t> &mask);
  // int32_t add(IVEImage *pSrc1, IVEImage *pSrc2, IVEImage *pDst,
  //             float alpha = 1.0, float beta = 1.0);
  // int32_t add(IVEImage *pSrc1, IVEImage *pSrc2, IVEImage *pDst,
  //             unsigned short alpha = std::numeric_limits<unsigned
  //             short>::max(), unsigned short beta =
  //             std::numeric_limits<unsigned short>::max());
  // int32_t fillConst(IVEImage *pSrc, float value);
  // int32_t thresh(IVEImage *pSrc, IVEImage *pDst, ThreshMode mode,
  //                CVI_U8 u8LowThr, CVI_U8 u8HighThr, CVI_U8 u8MinVal,
  //                CVI_U8 u8MidVal, CVI_U8 u8MaxVal);
  int32_t frameDiff(IVEImage *pSrc1, IVEImage *pSrc2, IVEImage *pDst,
                    CVI_U8 threshold);
  IVEImpl *getImpl();

 private:
  std::shared_ptr<IVEImpl> mp_impl_;
};

}  // namespace ive
