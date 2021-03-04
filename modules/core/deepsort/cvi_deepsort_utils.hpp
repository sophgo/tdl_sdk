#ifndef _CVI_DEEPSORT_UTILS_HPP_
#define _CVI_DEEPSORT_UTILS_HPP_

#include "core/cviai_core.h"
#include "cvi_deepsort_types_internal.hpp"

BBOX bbox_tlwh2tlbr(const BBOX &bbox_tlwh);

BBOX bbox_tlwh2xyah(const BBOX &bbox_tlwh);

/* DEBUG CODE */
std::string get_INFO_Vector_Int(const std::vector<int> &idxes, int w = 5);
std::string get_INFO_Vector_Pair_Int_Int(const std::vector<std::pair<int, int>> &pairs, int w = 5);

std::string get_INFO_Match_Pair(const std::vector<std::pair<int, int>> &pairs,
                                const std::vector<int> &idxes, int w = 5);

#endif /* _CVI_DEEPSORT_UTILS_HPP_ */