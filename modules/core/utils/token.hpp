#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "core/object/cvtdl_object_types.h"

namespace cvitdl {
int token_bpe(const std::string& encoderFile, const std::string& bpeFile,
              const std::string& textFile, std::vector<std::vector<int32_t>>& tokens);
}  // namespace cvitdl

class WordPieceTokenizer {
 public:
  WordPieceTokenizer(const std::string& vocabFile);
  ~WordPieceTokenizer();

  int tokenize(const std::string& textFile, cvtdl_tokens* tokens);
  int decode(cvtdl_tokens* tokens);

 private:
  int max_input_chars_per_word = 15;
  void LoadVocab(const std::string& fp);
  std::vector<std::string> splitString(const std::string& str);
  std::vector<int32_t> wordPieceTokenize(const std::string& word);
  std::unordered_map<std::string, uint32_t> vocab;
  std::unordered_map<uint32_t, std::string> decode_vocab;
};