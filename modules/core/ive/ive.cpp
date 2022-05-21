#include "ive.hpp"
#include "impl_ive.hpp"

namespace ive {
IVEImage::IVEImage() : mpImpl(IVEImageImpl::create()) {}

IVEImage::~IVEImage() {}

CVI_S32 IVEImage::toFrame(VIDEO_FRAME_INFO_S *frame, bool invertPackage) {
  return mpImpl->toFrame(frame, invertPackage);
}

CVI_S32 IVEImage::fromFrame(VIDEO_FRAME_INFO_S *frame) { return mpImpl->fromFrame(frame); }

CVI_S32 IVEImage::bufFlush() { return mpImpl->bufFlush(); }

CVI_S32 IVEImage::bufRequest() { return mpImpl->bufRequest(); }

CVI_S32 IVEImage::create(IVE *ive_instance, ImageType enType, CVI_U16 u16Width, CVI_U16 u16Height) {
  return mpImpl->create(ive_instance->getImpl(), enType, u16Width, u16Height);
}

CVI_S32 IVEImage::create(IVE *ive_instance, ImageType enType, CVI_U16 u16Width, CVI_U16 u16Height,
                         IVEImage *buf) {
  return mpImpl->create(ive_instance->getImpl(), enType, u16Width, u16Height, buf->getImpl());
}

IVEImageImpl *IVEImage::getImpl() { return mpImpl.get(); }

CVI_U32 IVEImage::getHeight() { return mpImpl->getHeight(); }

CVI_U32 IVEImage::getWidth() { return mpImpl->getWidth(); }

ImageType IVEImage::getType() { return mpImpl->getType(); }

std::vector<CVI_U32> IVEImage::getStride() { return mpImpl->getStride(); }

std::vector<CVI_U8 *> IVEImage::getVAddr() { return mpImpl->getVAddr(); }

std::vector<CVI_U64> IVEImage::getPAddr() { return mpImpl->getPAddr(); }

CVI_S32 IVEImage::free() { return mpImpl->free(); }

CVI_S32 IVEImage::write(const std::string &fname) { return mpImpl->write(fname); }

IVE::IVE() : mpImpl(IVEImpl::create()) {}

IVE::~IVE() {}

CVI_S32 IVE::init() { return mpImpl->init(); }

CVI_S32 IVE::destroy() { return mpImpl->destroy(); }

IVEImpl *IVE::getImpl() { return mpImpl.get(); }

CVI_S32 IVE::fillConst(IVEImage *pSrc, float value) {
  return mpImpl->fillConst(pSrc->getImpl(), value);
}

CVI_S32 IVE::dma(IVEImage *pSrc, IVEImage *pDst, DMAMode mode, CVI_U64 u64Val, CVI_U8 u8HorSegSize,
                 CVI_U8 u8ElemSize, CVI_U8 u8VerSegRows) {
  return mpImpl->dma(pSrc->getImpl(), pDst->getImpl(), mode, u64Val, u8HorSegSize, u8ElemSize,
                     u8VerSegRows);
}

CVI_S32 IVE::sub(IVEImage *pSrc1, IVEImage *pSrc2, IVEImage *pDst, SubMode mode) {
  return mpImpl->sub(pSrc1->getImpl(), pSrc2->getImpl(), pDst->getImpl(), mode);
}

CVI_S32 IVE::andImage(IVEImage *pSrc1, IVEImage *pSrc2, IVEImage *pDst) {
  return mpImpl->andImage(pSrc1->getImpl(), pSrc2->getImpl(), pDst->getImpl());
}

CVI_S32 IVE::orImage(IVEImage *pSrc1, IVEImage *pSrc2, IVEImage *pDst) {
  return mpImpl->orImage(pSrc1->getImpl(), pSrc2->getImpl(), pDst->getImpl());
}

CVI_S32 IVE::erode(IVEImage *pSrc1, IVEImage *pDst, const std::vector<CVI_S32> &mask) {
  return mpImpl->erode(pSrc1->getImpl(), pDst->getImpl(), mask);
}

CVI_S32 IVE::dilate(IVEImage *pSrc1, IVEImage *pDst, const std::vector<CVI_S32> &mask) {
  return mpImpl->dilate(pSrc1->getImpl(), pDst->getImpl(), mask);
}

CVI_S32 IVE::add(IVEImage *pSrc1, IVEImage *pSrc2, IVEImage *pDst, float alpha, float beta) {
  return mpImpl->add(pSrc1->getImpl(), pSrc2->getImpl(), pDst->getImpl(), alpha, beta);
}

CVI_S32 IVE::add(IVEImage *pSrc1, IVEImage *pSrc2, IVEImage *pDst, unsigned short alpha,
                 unsigned short beta) {
  return mpImpl->add(pSrc1->getImpl(), pSrc2->getImpl(), pDst->getImpl(), alpha, beta);
}

CVI_S32 IVE::thresh(IVEImage *pSrc, IVEImage *pDst, ThreshMode mode, CVI_U8 u8LowThr,
                    CVI_U8 u8HighThr, CVI_U8 u8MinVal, CVI_U8 u8MidVal, CVI_U8 u8MaxVal) {
  return mpImpl->thresh(pSrc->getImpl(), pDst->getImpl(), mode, u8LowThr, u8HighThr, u8MinVal,
                        u8MidVal, u8MaxVal);
}
}  // namespace ive