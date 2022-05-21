#pragma once

#include <cvi_comm.h>
#include <limits>
#include <memory>
#include <vector>
#include "ive_common.hpp"

namespace ive {
class IVEImpl;

// abstract class for implmentation
class IVEImageImpl {
 public:
  IVEImageImpl() = default;
  virtual ~IVEImageImpl() = default;
  static IVEImageImpl *create();

  virtual CVI_S32 toFrame(VIDEO_FRAME_INFO_S *frame, bool invertPackage = false) = 0;
  virtual CVI_S32 fromFrame(VIDEO_FRAME_INFO_S *frame) = 0;
  virtual CVI_S32 bufFlush() = 0;
  virtual CVI_S32 bufRequest() = 0;
  virtual CVI_S32 create(IVEImpl *ive_instance, ImageType enType, CVI_U16 u16Width,
                         CVI_U16 u16Height) = 0;
  virtual CVI_S32 create(IVEImpl *ive_instance, ImageType enType, CVI_U16 u16Width,
                         CVI_U16 u16Height, IVEImageImpl *buf) = 0;
  virtual CVI_S32 free() = 0;

  virtual CVI_U32 getHeight() = 0;
  virtual CVI_U32 getWidth() = 0;
  virtual std::vector<CVI_U32> getStride() = 0;
  virtual std::vector<CVI_U8 *> getVAddr() = 0;
  virtual std::vector<CVI_U64> getPAddr() = 0;
  virtual ImageType getType() = 0;
  virtual CVI_S32 write(const std::string &fname) = 0;

  // TODO: Maybe there are more elegant ways to get handle
  virtual void *getHandle() = 0;
};

// abstract class for implmentation
class IVEImpl {
 public:
  IVEImpl() = default;
  virtual ~IVEImpl() = default;
  static IVEImpl *create();

  virtual CVI_S32 init() = 0;
  virtual CVI_S32 destroy() = 0;

  virtual CVI_S32 fillConst(IVEImageImpl *pSrc, float value) = 0;
  virtual CVI_S32 dma(IVEImageImpl *pSrc, IVEImageImpl *pDst, DMAMode mode = DIRECT_COPY,
                      CVI_U64 u64Val = 0, CVI_U8 u8HorSegSize = 0, CVI_U8 u8ElemSize = 0,
                      CVI_U8 u8VerSegRows = 0) = 0;
  virtual CVI_S32 sub(IVEImageImpl *pSrc1, IVEImageImpl *pSrc2, IVEImageImpl *pDst,
                      SubMode mode = ABS) = 0;
  virtual CVI_S32 andImage(IVEImageImpl *pSrc1, IVEImageImpl *pSrc2, IVEImageImpl *pDst) = 0;
  virtual CVI_S32 orImage(IVEImageImpl *pSrc1, IVEImageImpl *pSrc2, IVEImageImpl *pDst) = 0;
  virtual CVI_S32 erode(IVEImageImpl *pSrc1, IVEImageImpl *pDst,
                        const std::vector<CVI_S32> &mask) = 0;
  virtual CVI_S32 dilate(IVEImageImpl *pSrc1, IVEImageImpl *pDst,
                         const std::vector<CVI_S32> &mask) = 0;
  virtual CVI_S32 add(IVEImageImpl *pSrc1, IVEImageImpl *pSrc2, IVEImageImpl *pDst,
                      float alpha = 1.0, float beta = 1.0) = 0;
  virtual CVI_S32 add(IVEImageImpl *pSrc1, IVEImageImpl *pSrc2, IVEImageImpl *pDst,
                      unsigned short alpha = std::numeric_limits<unsigned short>::max(),
                      unsigned short beta = std::numeric_limits<unsigned short>::max()) = 0;
  virtual CVI_S32 thresh(IVEImageImpl *pSrc, IVEImageImpl *pDst, ThreshMode mode, CVI_U8 u8LowThr,
                         CVI_U8 u8HighThr, CVI_U8 u8MinVal, CVI_U8 u8MidVal, CVI_U8 u8MaxVal) = 0;

  // TODO: Maybe there are more elegant ways to get handle
  virtual void *getHandle() = 0;
};

}  // namespace ive