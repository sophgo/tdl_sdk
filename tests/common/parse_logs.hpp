#pragma once
#include <algorithm>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

#include <experimental/filesystem>
#include <iomanip>
// 注意：fs 的别名与头文件包含放在 .cpp 中处理，这里不做包含以避免冲突。

struct Record;

// 带参数的 summary，便于从 test_runner 或独立程序调用
void summary(const std::string &log_dir, const std::string &output_md);
// 兼容的无参版本（默认 ./log -> summary.md）
void summary();