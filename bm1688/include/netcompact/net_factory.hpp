#ifndef INCLUDE_NET_FACTORY_H_
#define INCLUDE_NET_FACTORY_H_

#include "netcompact/net/bm1688net.hpp"

namespace nncompact {

class NetFactory {
 public:
  NetFactory() {}

  ~NetFactory() {}
  enum NetType { BM1684 = 0 };

 public:
  std::shared_ptr<Net> create_net(NetType type, const stNetParam &param) {
    switch (type) {
      case BM1684:
        return std::make_shared<BM1688Net>(param);

      default:
        break;
    }
    return NULL;
  }
};
}  // namespace nncompact

#endif
