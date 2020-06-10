// Copyright 2018 Bitmain Inc.
// License
// Author Tim Ho <tim.ho@bitmain.com>
/**
 * @file string_utils.hpp
 * @ingroup NETWORKS_UTILS
 */
#pragma once

#include <string>

namespace qnn {
namespace utils {

bool FindString(const std::string& source, const std::string& target);

}  // namespace utils
}  // namespace qnn
