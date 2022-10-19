#ifndef FILE_HEAD_SYS_UTILS_HPP
#define FILE_HEAD_SYS_UTILS_HPP
#include <iostream>
#include <string>
#include <vector>
using namespace std;
char *get_framelist_path(const char *in_path);
char *get_frame_path(const char *in_path);
char *get_out_path(const char *out_path, const char *frame_name);
std::vector<std::string> read_file_lines(std::string strfile);
std::string replace_file_ext(const std::string &src_file_name, const std::string &new_file_ext);
std::string get_directory_name(const std::string &str_path);
std::string join_path(const std::string &str_path_parent, const std::string &str_path_sub);
bool create_directory(const std::string &str_dir);
std::vector<std::string> getImgList(std::string dir_path);

#endif