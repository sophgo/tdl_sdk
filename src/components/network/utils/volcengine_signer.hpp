#pragma once
#include <map>
#include <string>
#include <vector>

class VolcengineSigner {
 public:
  VolcengineSigner(const std::string &ak, const std::string &sk,
                   const std::string &region);

  std::string Sign(const std::string &Method,
                   const std::map<std::string, std::string> &QueryString,
                   const std::map<std::string, std::string> &Headers,
                   const std::string &RequestPayload) const;
  static std::vector<unsigned char> Hash(const std::string &input);
  static std::string HexEncode(const std::vector<unsigned char> &data);
  static std::string Hmac(const std::string &key, const std::string &message);

 private:
  std::string ak_;
  std::string sk_;
  std::string region_;
};