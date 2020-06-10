// Copyright 2018 Bitmain Inc.
// License
// Author
/**
 * @file file_utils.hpp
 * @ingroup NETWORKS_FS
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace qnn {
namespace utils {

extern std::string GetBasename(char c, const std::string &src);
extern std::string GetFilePostfix(char *path);
extern std::string GetDirName(const std::string &src);
extern std::string ReplaceExt(const std::string &ext, const std::string &src);
extern std::string ReplaceFilename(const std::string &file, const std::string &src);
extern std::string GetFilename(const std::string &src);
extern std::string AddFilenamePrefix(const std::string &src, const std::string &prefix);
extern void DumpDataToFile(const std::string &name, char *data, int size);
extern void StringSplit(const std::string &src, std::vector<std::string> *results,
                        const std::string &c);
extern std::unique_ptr<char[]> LoadDataFromFile(const std::string &name, int *len);
extern bool FileExist(const std::string &file_path);
extern bool DirExist(const std::string &dir_path);
extern int FindPic(const std::string &target_dir, std::vector<std::string> &result);

}  // namespace utils
}  // namespace qnn
