#include <cvi_comm_video.h>

namespace cviai {

class VpssEngine {
 public:
  VpssEngine();
  ~VpssEngine();
  int init(bool enable_log);
  int stop();
  VPSS_GRP getGrpId();
  CVI_BOOL* const getEnabledChn();

 private:
  void enableLog();

  bool m_enable_log = false;
  bool m_is_vpss_init = false;
  VPSS_GRP m_grpid = -1;
  int m_enabled_chn_num = 2;
  CVI_BOOL m_chn_enable[VPSS_MAX_PHY_CHN_NUM] = {0};
};
}  // namespace cviai