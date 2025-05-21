#ifndef __IMPL_IVE_HPP__
#define __IMPL_IVE_HPP__

#include <limits>
#include <memory>
#include <vector>

#include "cvi_comm.hpp"
#include "ive_common.hpp"

namespace ive {
class IVEImpl;

// abstract class for implmentation
class IVEImageImpl {
 public:
  IVEImageImpl() = default;
  virtual ~IVEImageImpl() = default;
  static IVEImageImpl *create();

  // virtual int32_t toFrame(VIDEO_FRAME_INFO_S *frame,
  //                         bool invertPackage = false) = 0;
  virtual int32_t fromFrame(VIDEO_FRAME_INFO_S *frame) = 0;
  virtual int32_t bufFlush(IVEImpl *ive_instance) = 0;
  virtual int32_t bufRequest(IVEImpl *ive_instance) = 0;
  virtual int32_t create(IVEImpl *ive_instance, ImageType enType,
                         CVI_U16 u16Width, CVI_U16 u16Height, bool cached) = 0;
  virtual int32_t create(IVEImpl *ive_instance, ImageType enType,
                         CVI_U16 u16Width, CVI_U16 u16Height, IVEImageImpl *buf,
                         bool cached) = 0;
  virtual int32_t create(IVEImpl *ive_instance) = 0;
  virtual int32_t free() = 0;

  virtual CVI_U32 getHeight() = 0;
  virtual CVI_U32 getWidth() = 0;
  virtual std::vector<CVI_U32> getStride() = 0;
  virtual std::vector<CVI_U8 *> getVAddr() = 0;
  virtual std::vector<CVI_U64> getPAddr() = 0;
  virtual ImageType getType() = 0;
  virtual int32_t write(const std::string &fname) = 0;

  // TODO: Maybe there are more elegant ways to get handle
  virtual void *getHandle() = 0;
};

// abstract class for implmentation
class IVEImpl {
 public:
  IVEImpl() = default;
  virtual ~IVEImpl() = default;
  static IVEImpl *create();

  uint32_t getAlignedWidth(uint32_t width) {
    uint32_t align = getWidthAlign();
    uint32_t stride = (uint32_t)(width / align) * align;
    if (stride < width) {
      stride += align;
    }
    return stride;
  }

  virtual int32_t init() = 0;
  virtual int32_t destroy() = 0;
  virtual CVI_U32 getWidthAlign() = 0;
  // virtual int32_t fillConst(IVEImageImpl *pSrc, float value) = 0;
  virtual int32_t dma(IVEImageImpl *pSrc, IVEImageImpl *pDst,
                      DMAMode mode = DIRECT_COPY, CVI_U64 u64Val = 0,
                      CVI_U8 u8HorSegSize = 0, CVI_U8 u8ElemSize = 0,
                      CVI_U8 u8VerSegRows = 0) = 0;
  // virtual int32_t sub(IVEImageImpl *pSrc1, IVEImageImpl *pSrc2,
  //                     IVEImageImpl *pDst, SubMode mode = ABS) = 0;
  // virtual int32_t roi(IVEImageImpl *pSrc, IVEImageImpl *pDst, uint32_t x1,
  //                     uint32_t x2, uint32_t y1, uint32_t y2) = 0;
  // virtual int32_t andImage(IVEImageImpl *pSrc1, IVEImageImpl *pSrc2,
  //                          IVEImageImpl *pDst) = 0;
  // virtual int32_t orImage(IVEImageImpl *pSrc1, IVEImageImpl *pSrc2,
  //                         IVEImageImpl *pDst) = 0;
  // virtual int32_t erode(IVEImageImpl *pSrc1, IVEImageImpl *pDst,
  //                       const std::vector<int32_t> &mask) = 0;
  // virtual int32_t dilate(IVEImageImpl *pSrc1, IVEImageImpl *pDst,
  //                        const std::vector<int32_t> &mask) = 0;
  // virtual int32_t add(IVEImageImpl *pSrc1, IVEImageImpl *pSrc2,
  //                     IVEImageImpl *pDst, float alpha = 1.0,
  //                     float beta = 1.0) = 0;
  // virtual int32_t add(
  //     IVEImageImpl *pSrc1, IVEImageImpl *pSrc2, IVEImageImpl *pDst,
  //     unsigned short alpha = std::numeric_limits<unsigned short>::max(),
  //     unsigned short beta = std::numeric_limits<unsigned short>::max()) = 0;
  // virtual int32_t thresh(IVEImageImpl *pSrc, IVEImageImpl *pDst,
  //                        ThreshMode mode, CVI_U8 u8LowThr, CVI_U8 u8HighThr,
  //                        CVI_U8 u8MinVal, CVI_U8 u8MidVal, CVI_U8 u8MaxVal) =
  //                        0;
  virtual int32_t frameDiff(IVEImageImpl *pSrc1, IVEImageImpl *pSrc2,
                            IVEImageImpl *pDst, CVI_U8 threshold) = 0;

  // TODO: Maybe there are more elegant ways to get handle
  virtual void *getHandle() = 0;
};

}  // namespace ive
#endif  // __IMPL_IVE_HPP__