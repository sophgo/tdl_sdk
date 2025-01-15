// #include "bm168x_net.hpp"
// #include "cmodel_net.hpp"
#ifdef __SOPHON__
#pragma message("__SOPHON__ is defined at compile time")
#endif

#ifdef __SOPHON__
#include "net/bm168x_net.hpp"
#else
#include "net/cvi_net.hpp"
#endif
#include "cvi_tdl_log.hpp"

std::shared_ptr<BaseNet> NetFactory::createNet(const NetParam& net_param,
                                               InferencePlatform platform) {
  switch (platform) {
    case InferencePlatform::CVITEK:
#if !defined(__SOPHON__) && !defined(__CV186X__)
      return std::make_shared<CviNet>(net_param);
#else
      return nullptr;
#endif
    case InferencePlatform::BM168X:
#ifdef __SOPHON__
      LOGI("create BM168xNet");
      return std::make_shared<BM168xNet>(net_param);
#else
      return nullptr;
#endif
    // case InferencePlatform::CMODEL:
    //   return std::make_shared<CmodelNet>(net_param);
    default:
      LOGE("unknown platform %d", platform);
      return nullptr;
  }
}
