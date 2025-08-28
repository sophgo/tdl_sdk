#pragma once
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>
#include "utils/tdl_log.hpp"

namespace cvitdl {
namespace unitest {

std::string gen_model_suffix();
std::string gen_platform();
std::string gen_model_dir();
std::vector<std::string> get_platform_list();
std::string extractModelIdFromName(const std::string& model_name);
std::map<std::string, float> getCustomRegressionConfig(
    const std::string& model_name);

}  // namespace unitest
}  // namespace cvitdl