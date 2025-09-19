// #include "bm168x_net.hpp"
// #include "cmodel_net.hpp"

#if defined(__BM168X__) || defined(__CV186X__) || defined(__CV184X__) || \
    defined(__CMODEL_CV184X__)
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
    case InferencePlatform::CMODEL_CV181X:
#if !defined(__BM168X__) && !defined(__CV186X__) && !defined(__CV184X__) && \
    !defined(__CMODEL_CV184X__)
      return std::make_shared<CviNet>(net_param);
#else
      return nullptr;
#endif
    case InferencePlatform::BM168X:
    case InferencePlatform::CV186X:
    case InferencePlatform::CV184X:
    case InferencePlatform::CMODEL_CV184X:
#if defined(__BM168X__) || defined(__CV186X__) || defined(__CV184X__) || \
    defined(__CMODEL_CV184X__)
      LOGI("create BM168xNet");
      return std::make_shared<BM168xNet>(net_param);
#else
      return nullptr;
#endif
    default:
      LOGE("unknown platform %d", static_cast<int>(platform));
      return nullptr;
  }
}
