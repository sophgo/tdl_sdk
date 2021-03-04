#include "cvi_deepsort_utils.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>

BBOX bbox_tlwh2tlbr(const BBOX &bbox_tlwh) {
  BBOX bbox_tlbr;
  bbox_tlbr(0) = bbox_tlwh(0);
  bbox_tlbr(1) = bbox_tlwh(1);
  bbox_tlbr(2) = bbox_tlwh(0) + bbox_tlwh(2);
  bbox_tlbr(3) = bbox_tlwh(1) + bbox_tlwh(3);
  return bbox_tlbr;
}

BBOX bbox_tlwh2xyah(const BBOX &bbox_tlwh) {
  BBOX bbox_xyah;
  bbox_xyah(0) = bbox_tlwh(0) + 0.5 * bbox_tlwh(2);
  bbox_xyah(1) = bbox_tlwh(1) + 0.5 * bbox_tlwh(3);
  bbox_xyah(2) = bbox_tlwh(2) / bbox_tlwh(3);
  bbox_xyah(3) = bbox_tlwh(3);
  return bbox_xyah;
}

std::string get_INFO_Vector_Int(const std::vector<int> &idxes, int w) {
  if (idxes.empty()) {
    return std::string("[]");
  }
  std::stringstream ss;
  ss << "[" << std::setw(w) << idxes[0];
  for (size_t i = 1; i < idxes.size(); i++) {
    ss << "," << std::setw(w) << idxes[i];
  }
  ss << "]";
  return std::string(ss.str());
}

std::string get_INFO_Vector_Pair_Int_Int(const std::vector<std::pair<int, int>> &pairs, int w) {
  if (pairs.empty()) {
    return std::string("");
  }
  std::stringstream ss;
  for (size_t i = 0; i < pairs.size(); i++) {
    ss << "<" << std::setw(w) << pairs[i].first << "," << std::setw(w) << pairs[i].second << ">\n";
  }
  return std::string(ss.str());
}

std::string get_INFO_Match_Pair(const std::vector<std::pair<int, int>> &pairs,
                                const std::vector<int> &idxes, int w) {
  if (pairs.empty()) {
    return std::string("");
  }
  std::stringstream ss;
  for (size_t i = 0; i < pairs.size(); i++) {
    ss << "<<" << std::setw(w) << idxes[pairs[i].first] << ">" << std::setw(w) << pairs[i].first
       << "," << std::setw(w) << pairs[i].second << ">\n";
  }
  return std::string(ss.str());
}
