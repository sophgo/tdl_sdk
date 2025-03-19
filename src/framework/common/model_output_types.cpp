#include "common/model_output_types.hpp"

ModelFeatureInfo::~ModelFeatureInfo() {
  if (embedding) {
    delete[] embedding;
    embedding = nullptr;
  }
}

ModelClipFeatureInfo::~ModelClipFeatureInfo() {
  if (embedding) {
    delete[] embedding;
    embedding = nullptr;
  }
}

ModelOcrInfo::~ModelOcrInfo() {
  if (text_info) {
    delete[] text_info;
    text_info = nullptr;
    length = 0;
  }
}