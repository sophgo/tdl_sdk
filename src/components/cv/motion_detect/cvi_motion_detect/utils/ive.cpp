#include <cstring>

#include "impl_ive.hpp"
#include "ive.hpp"

namespace ive {
IVEImage::IVEImage() : mp_impl_(IVEImageImpl::create()) {}

IVEImage::~IVEImage() {}

// int32_t IVEImage::toFrame(VIDEO_FRAME_INFO_S *frame, bool invertPackage) {
//   return mp_impl_->toFrame(frame, invertPackage);
// }

int32_t IVEImage::fromFrame(VIDEO_FRAME_INFO_S *frame) {
  return mp_impl_->fromFrame(frame);
}

int32_t IVEImage::bufFlush(IVE *ive_instance) {
  return mp_impl_->bufFlush(ive_instance->getImpl());
}

int32_t IVEImage::bufRequest(IVE *ive_instance) {
  return mp_impl_->bufRequest(ive_instance->getImpl());
}

int32_t IVEImage::create(IVE *ive_instance, ImageType enType, CVI_U16 u16Width,
                         CVI_U16 u16Height, bool cached) {
  return mp_impl_->create(ive_instance->getImpl(), enType, u16Width, u16Height,
                          cached);
}

int32_t IVEImage::create(IVE *ive_instance, ImageType enType, CVI_U16 u16Width,
                         CVI_U16 u16Height, IVEImage *buf, bool cached) {
  return mp_impl_->create(ive_instance->getImpl(), enType, u16Width, u16Height,
                          buf->getImpl(), cached);
}

int32_t IVEImage::create(IVE *ive_instance) {
  return mp_impl_->create(ive_instance->getImpl());
}

IVEImageImpl *IVEImage::getImpl() { return mp_impl_.get(); }

CVI_U32 IVEImage::getHeight() { return mp_impl_->getHeight(); }

CVI_U32 IVEImage::getWidth() { return mp_impl_->getWidth(); }

ImageType IVEImage::getType() { return mp_impl_->getType(); }

std::vector<CVI_U32> IVEImage::getStride() { return mp_impl_->getStride(); }

std::vector<CVI_U8 *> IVEImage::getVAddr() { return mp_impl_->getVAddr(); }

std::vector<CVI_U64> IVEImage::getPAddr() { return mp_impl_->getPAddr(); }

int32_t IVEImage::setZero(IVE *ive_instance) {
  std::vector<CVI_U8 *> v_addrs = getVAddr();
  std::vector<CVI_U32> strides = getStride();

  if (v_addrs.size() != strides.size()) {
    // LOGE("vaddrs num:%d,strides
    // num:%d\n",(int)v_addrs.size(),(int)strides.size());
    return CVI_FAILURE;
  }
  CVI_U32 imh = getHeight();
  for (uint32_t i = 0; i < v_addrs.size(); i++) {
    memset(v_addrs[i], 0, imh * strides[i]);
  }
  return bufFlush(ive_instance);
}

int32_t IVEImage::free() { return mp_impl_->free(); }

int32_t IVEImage::write(const std::string &fname) {
  return mp_impl_->write(fname);
}

IVE::IVE() : mp_impl_(IVEImpl::create()) {}

IVE::~IVE() {}

int32_t IVE::init() { return mp_impl_->init(); }

int32_t IVE::destroy() { return mp_impl_->destroy(); }

IVEImpl *IVE::getImpl() { return mp_impl_.get(); }

CVI_U32 IVE::getAlignedWidth(uint32_t width) {
  return mp_impl_->getAlignedWidth(width);
}

// int32_t IVE::fillConst(IVEImage *pSrc, float value) {
//   return mp_impl_->fillConst(pSrc->getImpl(), value);
// }

int32_t IVE::dma(IVEImage *pSrc, IVEImage *pDst, DMAMode mode, CVI_U64 u64Val,
                 CVI_U8 u8HorSegSize, CVI_U8 u8ElemSize, CVI_U8 u8VerSegRows) {
  return mp_impl_->dma(pSrc->getImpl(), pDst->getImpl(), mode, u64Val,
                       u8HorSegSize, u8ElemSize, u8VerSegRows);
}

// int32_t IVE::sub(IVEImage *pSrc1, IVEImage *pSrc2, IVEImage *pDst,
//                  SubMode mode) {
//   return mp_impl_->sub(pSrc1->getImpl(), pSrc2->getImpl(), pDst->getImpl(),
//                        mode);
// }

// int32_t IVE::roi(IVEImage *pSrc, IVEImage *pDst, uint32_t x1, uint32_t x2,
// uint32_t y1,
//                  uint32_t y2) {
//   return mp_impl_->roi(pSrc->getImpl(), pDst->getImpl(), x1, x2, y1, y2);
// }

// int32_t IVE::andImage(IVEImage *pSrc1, IVEImage *pSrc2, IVEImage *pDst) {
//   return mp_impl_->andImage(pSrc1->getImpl(), pSrc2->getImpl(),
//   pDst->getImpl());
// }

// int32_t IVE::orImage(IVEImage *pSrc1, IVEImage *pSrc2, IVEImage *pDst) {
//   return mp_impl_->orImage(pSrc1->getImpl(), pSrc2->getImpl(),
//   pDst->getImpl());
// }

// int32_t IVE::erode(IVEImage *pSrc1, IVEImage *pDst, const
// std::vector<int32_t> &mask) {
//   return mp_impl_->erode(pSrc1->getImpl(), pDst->getImpl(), mask);
// }

// int32_t IVE::dilate(IVEImage *pSrc1, IVEImage *pDst,
//                     const std::vector<int32_t> &mask) {
//   return mp_impl_->dilate(pSrc1->getImpl(), pDst->getImpl(), mask);
// }

// int32_t IVE::add(IVEImage *pSrc1, IVEImage *pSrc2, IVEImage *pDst, float
// alpha, float beta) {
//   return mp_impl_->add(pSrc1->getImpl(), pSrc2->getImpl(), pDst->getImpl(),
//   alpha, beta);
// }

// int32_t IVE::add(IVEImage *pSrc1, IVEImage *pSrc2, IVEImage *pDst, unsigned
// short alpha,
//                  unsigned short beta) {
//   return mp_impl_->add(pSrc1->getImpl(), pSrc2->getImpl(), pDst->getImpl(),
//   alpha, beta);
// }

// int32_t IVE::thresh(IVEImage *pSrc, IVEImage *pDst, ThreshMode mode,
//                     CVI_U8 u8LowThr, CVI_U8 u8HighThr, CVI_U8 u8MinVal,
//                     CVI_U8 u8MidVal, CVI_U8 u8MaxVal) {
//   return mp_impl_->thresh(pSrc->getImpl(), pDst->getImpl(), mode, u8LowThr,
//                           u8HighThr, u8MinVal, u8MidVal, u8MaxVal);
// }

int32_t IVE::frameDiff(IVEImage *pSrc1, IVEImage *pSrc2, IVEImage *pDst,
                       CVI_U8 threshold) {
  return mp_impl_->frameDiff(pSrc1->getImpl(), pSrc2->getImpl(),
                             pDst->getImpl(), threshold);
}

}  // namespace ive
