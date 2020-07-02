#pragma once

#include <string>

namespace utils {
namespace file {

bool exists(const std::string &path);
bool is_regular(const std::string &path);
bool is_directory(const std::string &path);
std::string content(const std::string &path, size_t offset, size_t length);
bool remove(const std::string &path);
std::string directory_name(const std::string &path);
bool mkdir(const std::string &path);

}
}