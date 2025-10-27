#include "volcengine_signer.hpp"
#include <openssl/hmac.h>
#include <openssl/sha.h>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>

VolcengineSigner::VolcengineSigner(const std::string &ak, const std::string &sk,
                                   const std::string &region)
    : ak_(ak), sk_(sk), region_(region) {}

std::vector<unsigned char> VolcengineSigner::Hash(const std::string &input) {
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char *>(input.data()), input.size(),
         hash);
  return std::vector<unsigned char>(hash, hash + SHA256_DIGEST_LENGTH);
}

std::string VolcengineSigner::HexEncode(
    const std::vector<unsigned char> &data) {
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (unsigned char c : data) {
    oss << std::setw(2) << static_cast<int>(c);
  }
  return oss.str();
}

std::string VolcengineSigner::Hmac(const std::string &key,
                                   const std::string &message) {
  try {
    std::vector<unsigned char> keyBytes;
    if (key.length() % 2 == 0) {
      keyBytes.resize(key.length() / 2);
      for (size_t i = 0; i < keyBytes.size(); ++i) {
        keyBytes[i] = static_cast<unsigned char>(
            std::stoi(key.substr(i * 2, 2), nullptr, 16));
      }
    } else {
      keyBytes.assign(key.begin(), key.end());
    }

    unsigned int len = 0;
    unsigned char *res =
        HMAC(EVP_sha256(), keyBytes.data(), static_cast<int>(keyBytes.size()),
             reinterpret_cast<const unsigned char *>(message.data()),
             message.size(), nullptr, &len);
    return HexEncode(std::vector<unsigned char>(res, res + len));
  } catch (...) {
    std::vector<unsigned char> keyBytes(key.begin(), key.end());
    unsigned int len = 0;
    unsigned char *res =
        HMAC(EVP_sha256(), keyBytes.data(), static_cast<int>(keyBytes.size()),
             reinterpret_cast<const unsigned char *>(message.data()),
             message.size(), nullptr, &len);
    return HexEncode(std::vector<unsigned char>(res, res + len));
  }
}

std::string VolcengineSigner::Sign(
    const std::string &Method,
    const std::map<std::string, std::string> &QueryString,
    const std::map<std::string, std::string> &Headers,
    const std::string &RequestPayload) const {
  std::string HTTPRequestMethod = Method;
  std::string CanonicalURI = "/";

  // 构建 CanonicalQueryString
  std::ostringstream qoss;
  bool firstQuery = true;
  for (const auto &pair : QueryString) {
    if (!firstQuery) qoss << '&';
    qoss << pair.first << '=' << pair.second;
    firstQuery = false;
  }
  std::string CanonicalQueryString = qoss.str();

  // 构建 CanonicalHeaders 和 SignedHeaders
  std::ostringstream hoss, ssh;
  bool firstHeader = true;
  for (const auto &pair : Headers) {
    std::string key = pair.first;
    std::string value = pair.second;
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    value.erase(0, value.find_first_not_of(" \t"));
    value.erase(value.find_last_not_of(" \t") + 1);
    if (!firstHeader) {
      hoss << '\n';
      ssh << ';';
    }
    hoss << key << ':' << value;
    ssh << key;
    firstHeader = false;
  }
  std::string CanonicalHeaders = hoss.str();
  std::string SignedHeaders = ssh.str();

  // 构建 CanonicalRequest
  std::ostringstream cross;
  cross << HTTPRequestMethod << '\n'
        << CanonicalURI << '\n'
        << CanonicalQueryString << '\n'
        << CanonicalHeaders << '\n'
        << '\n'
        << SignedHeaders << '\n'
        << HexEncode(Hash(RequestPayload));
  std::string CanonicalRequest = cross.str();

  std::string Algorithm = "HMAC-SHA256";
  std::string RequestDate = Headers.at("x-date");
  std::string service = "cv";
  std::string CredentialScope =
      RequestDate.substr(0, 8) + "/" + region_ + "/" + service + "/request";

  // 构建 StringToSign
  std::ostringstream stoss;
  stoss << Algorithm << '\n'
        << RequestDate << '\n'
        << CredentialScope << '\n'
        << HexEncode(Hash(CanonicalRequest));
  std::string StringToSign = stoss.str();

  std::string kSecret = sk_;
  std::string kDate = Hmac(kSecret, RequestDate.substr(0, 8));
  std::string kRegion = Hmac(kDate, region_);
  std::string kService = Hmac(kRegion, service);
  std::string kSigning = Hmac(kService, "request");

  std::string Signature = Hmac(kSigning, StringToSign);

  std::string Credential = ak_ + "/" + CredentialScope;
  std::ostringstream auth;
  auth << Algorithm << " Credential=" << Credential
       << ", SignedHeaders=" << SignedHeaders << ", Signature=" << Signature;
  return auth.str();
}