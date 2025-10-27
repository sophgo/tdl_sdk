#include "utils/common_utils.hpp"
#include <dlfcn.h>
#include <libgen.h>
#include <limits.h>  // for PATH_MAX
#include <unistd.h>
#include <string>
#include "utils/tdl_log.hpp"
uint32_t CommonUtils::getDataTypeSize(TDLDataType data_type) {
  switch (data_type) {
    case TDLDataType::FP32:
      return 4;
    case TDLDataType::INT32:
      return 4;
    case TDLDataType::UINT32:
      return 4;
    case TDLDataType::FP16:
      return 2;
    case TDLDataType::BF16:
      return 2;
    case TDLDataType::INT16:
      return 2;
    case TDLDataType::UINT16:
      return 2;
    case TDLDataType::INT8:
      return 1;
    case TDLDataType::UINT8:
      return 1;
    default:
      return 0;
  }
}

InferencePlatform CommonUtils::getPlatform() {
#if defined(__BM168X__)
  return InferencePlatform::BM168X;
#elif defined(__CV186X__)
  return InferencePlatform::CV186X;
#elif defined(__CV184X__)
  return InferencePlatform::CV184X;
#elif defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__)
  return InferencePlatform::CVITEK;
#elif defined(__CMODEL_CV181X__)
  return InferencePlatform::CMODEL_CV181X;
#elif defined(__CMODEL_CV184X__)
  return InferencePlatform::CMODEL_CV184X;
#else
  return InferencePlatform::UNKOWN;
#endif
}

bool CommonUtils::readBinaryFile(const std::string &strf,
                                 std::vector<uint8_t> &buffer) {
  FILE *fp = fopen(strf.c_str(), "rb");
  if (fp == nullptr) {
    LOGE("read file failed,%s\n", strf.c_str());
    return false;
  }
  fseek(fp, 0, SEEK_END);
  int len = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  buffer.resize(len);
  fread(buffer.data(), len, 1, fp);
  fclose(fp);
  return true;
}

std::string CommonUtils::getLibraryPath() {
  Dl_info info{};
  // use the address of any function or static in your .so:
  if (dladdr((void *)&CommonUtils::getLibraryPath, &info) && info.dli_fname) {
    // info.dli_fname is the path as the dynamic loader saw it
    char resolved[PATH_MAX];
    if (realpath(info.dli_fname, resolved))
      return std::string(resolved);
    else
      return std::string(info.dli_fname);
  }
  return {};
}

std::string CommonUtils::getLibraryDir() {
  auto full = getLibraryPath();
  auto pos = full.find_last_of('/');
  return (pos != std::string::npos ? full.substr(0, pos) : std::string{});
}

std::string CommonUtils::getParentDir(const std::string &path) {
  auto pos = path.find_last_of('/');
  return (pos != std::string::npos ? path.substr(0, pos) : std::string{});
}

std::string CommonUtils::getExecutableDir() {
  char buffer[PATH_MAX];
  auto len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
  if (len == -1) {
    return "";
  }
  buffer[len] = '\0';
  char dir[PATH_MAX];
  strncpy(dir, buffer, PATH_MAX);
  return std::string(dirname(dir));
}