// #include "bm168x_net.hpp"
// #include "cmodel_net.hpp"

#if defined(__BM168X__) || defined(__CV186X__)
#include "net/bm168x_net.hpp"
#else
#include "net/cvi_net.hpp"
#endif
#include "utils/tdl_log.hpp"

std::shared_ptr<BaseNet> NetFactory::createNet(const NetParam& net_param,
                                               InferencePlatform platform) {
  LOGI("createNet,platform: %d", (int)platform);
  switch (platform) {
    case InferencePlatform::CVITEK:
    case InferencePlatform::CMODEL:
#if !defined(__BM168X__) && !defined(__CV186X__)
      return std::make_shared<CviNet>(net_param);
#else
      return nullptr;
#endif
    case InferencePlatform::BM168X:
    case InferencePlatform::CV186X:
#if defined(__BM168X__) || defined(__CV186X__)
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
