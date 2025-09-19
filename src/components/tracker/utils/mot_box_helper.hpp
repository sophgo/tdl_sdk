#ifndef MOT_BOX_HELPER_HPP
#define MOT_BOX_HELPER_HPP

#include "mot/mot_type_defs.hpp"

class MotBoxHelper {
 public:
  static DETECTBOX convertToXYAH(const ObjectBoxInfo &box_info);
  static float calculateIOUOnFirst(const ObjectBoxInfo &box1,
                                   const ObjectBoxInfo &box2);
  static float calculateIOU(const ObjectBoxInfo &box1,
                            const ObjectBoxInfo &box2);

  static bool isBboxCrowded(const std::vector<ObjectBoxInfo> &dets,
                            int check_idx, float expand_ratio);
  static float getInterArea(const ObjectBoxInfo &box1,
                            const ObjectBoxInfo &box2);
  static float computeBoxSim(const ObjectBoxInfo &box1,
                             const ObjectBoxInfo &box2);
  static float calObjectPairScore(ObjectBoxInfo boxa, ObjectBoxInfo boxb,
                                  TDLObjectType typea, TDLObjectType typeb);
};

#endif
