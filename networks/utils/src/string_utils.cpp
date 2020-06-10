// Copyright 2018 Bitmain Inc.
// License
// Author Tim Ho <tim.ho@bitmain.com>

#include "string_utils.hpp"
#include <algorithm>

namespace qnn {
namespace utils {

bool FindString(const std::string& s1, const std::string& s2) {
    auto it = std::search(s1.begin(), s1.end(), s2.begin(), s2.end(), [](char ch1, char ch2) {
        return std::toupper(ch1) == std::toupper(ch2);
    });
    return (it != s1.end());
}

}  // namespace utils
}  // namespace qnn