#include "common/model_output_types.hpp"

ModelFeatureInfo::~ModelFeatureInfo() {
  if (embedding) {
    delete[] embedding;
    embedding = nullptr;
  }
}