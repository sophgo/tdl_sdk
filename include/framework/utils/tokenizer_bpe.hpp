#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

class BytePairEncoder {
 public:
  BytePairEncoder(const std::string& encoderFile, const std::string& bpeFile)
      : encoderFile(encoderFile), bpeFile(bpeFile) {}

  int tokenizerBPE(const std::string& textFile,
                   std::vector<std::vector<int32_t>>& tokens) {
    readVocab();
    buildPairsMap();

    std::vector<std::string> text = readTextFile(textFile);
    if (text.empty()) {
      return 1;
    }

    tokens.resize(text.size());
    processSentences(tokens, text);

    return 0;
  }

 private:
  std::string encoderFile;
  std::string bpeFile;
  std::unordered_map<std::string, uint32_t> vocab;
  std::unordered_map<std::pair<std::string, std::string>, int, pair_hash>
      pairs_map;

  void readVocab() {
    std::ifstream file(encoderFile);
    if (!file) {
      std::cerr << "Cannot open vocabulary file " << encoderFile << std::endl;
      exit(EXIT_FAILURE);
    }
    std::cerr << "Loading vocabulary from " << encoderFile << " ..."
              << std::endl;
    std::string line;
    while (std::getline(file, line)) {
      size_t pos = line.find(": ");
      if (pos == std::string::npos) {
        std::cerr << "Invalid line format: " << line << std::endl;
        continue;
      }
      std::string word = line.substr(0, pos);
      int count = stoi(line.substr(pos + 2));
      vocab[word] = count;
    }
    std::cerr << "Read " << vocab.size() << " words from vocabulary file."
              << std::endl;
  }

  void buildPairsMap() {
    const int start_line = 2;
    const int end_line = 48895;
    pairs_map = createPairsMap(bpeFile, start_line, end_line);
  }

  std::unordered_map<std::pair<std::string, std::string>, int, pair_hash>
  createPairsMap(const std::string& filename, int start_line, int end_line) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error opening file." << std::endl;
      return {};
    }

    std::string line;
    std::vector<std::pair<std::string, std::string>> pairs;
    int line_count = 0;

    while (std::getline(file, line) && line_count < end_line) {
      ++line_count;
      if (line_count >= start_line) {
        std::istringstream iss(line);
        std::string word1, word2;
        iss >> word1 >> word2;
        pairs.emplace_back(word1, word2);
      }
    }

    file.close();

    std::unordered_map<std::pair<std::string, std::string>, int, pair_hash>
        pairs_map;
    int value = 0;
    for (const auto& pair : pairs) {
      pairs_map[pair] = value++;
    }

    return pairs_map;
  }

  std::vector<std::string> readTextFile(const std::string& textFile) {
    std::vector<std::string> text;
    std::ifstream file(textFile);

    if (file.is_open()) {
      std::string line;
      while (std::getline(file, line)) {
        text.push_back(line);
      }
      file.close();
    } else {
      std::cerr << "Unable to open file." << std::endl;
    }

    return text;
  }

  void processSentences(std::vector<std::vector<int32_t>>& tokens,
                        const std::vector<std::string>& text) {
    std::string special = "[@#$%^&*!]";
    std::regex pattern(
        special +
            R"(|'s|'t|'re|'ve|'m|'ll|'d|[[:alpha:]]+|[[:digit:]]|[^[:space:][:alpha:][:digit:]]+)",
        std::regex_constants::icase);

    for (int j = 0; j < text.size(); j++) {
      std::string line = text[j];
      std::transform(line.begin(), line.end(), line.begin(),
                     ::tolower);  // convert all to lowercase
      std::vector<std::string> words;

      std::sregex_iterator it(line.begin(), line.end(), pattern);
      std::sregex_iterator end;

      while (it != end) {
        std::smatch match = *it;
        words.push_back(match.str());
        ++it;
      }

      tokens[j].push_back(vocab["<start_of_text>"]);

      for (const auto& word : words) {
        processWord(word, tokens[j]);
      }

      tokens[j].push_back(vocab["<end_of_text>"]);
      padTokens(tokens[j]);
    }
  }

  std::vector<std::string> splitWord(const std::string& token) {
    std::vector<std::string> word;
    for (size_t i = 0; i < token.size() - 1; ++i) {
      word.push_back(token.substr(i, 1));
    }
    word.push_back(token.substr(token.size() - 1) + "</w>");
    return word;
  }

  void processWord(const std::string& word, std::vector<int32_t>& token) {
    std::vector<std::string> split_word = splitWord(word);
    std::set<std::pair<std::string, std::string>> pairs = getPairs(split_word);

    if (pairs.empty()) {
      std::string token_str = word + "</w>";
      token.push_back(vocab[token_str]);
      return;
    }

    mergePairs(split_word, pairs);
    for (const auto& word_split : split_word) {
      token.push_back(vocab[word_split]);
    }
  }

  std::set<std::pair<std::string, std::string>> getPairs(
      const std::vector<std::string>& word) {
    std::set<std::pair<std::string, std::string>> pairs;
    std::string prev_char = word[0];
    for (size_t i = 1; i < word.size(); ++i) {
      pairs.insert(std::make_pair(prev_char, word[i]));
      prev_char = word[i];
    }
    return pairs;
  }

  void mergePairs(std::vector<std::string>& word,
                  std::set<std::pair<std::string, std::string>>& pairs) {
    while (true) {
      std::pair<std::string, std::string> bigram;

      if (pairs.size() > 1) {
        bigram = *std::min_element(
            pairs.begin(), pairs.end(),
            [this](const std::pair<std::string, std::string>& pair1,
                   const std::pair<std::string, std::string>& pair2) {
              int pair1Value = pairs_map.find(pair1) != pairs_map.end()
                                   ? pairs_map.at(pair1)
                                   : std::numeric_limits<int>::max();
              int pair2Value = pairs_map.find(pair2) != pairs_map.end()
                                   ? pairs_map.at(pair2)
                                   : std::numeric_limits<int>::max();
              return pair1Value < pair2Value;
            });
      } else {
        bigram = *pairs.begin();
      }

      if (pairs_map.find(bigram) == pairs_map.end()) {
        break;
      }

      std::string first = bigram.first;
      std::string second = bigram.second;
      std::vector<std::string> new_word;
      size_t i = 0;

      while (i < word.size()) {
        auto it = std::find(word.begin() + i, word.end(), first);
        if (it == word.end()) {
          new_word.insert(new_word.end(), word.begin() + i, word.end());
          break;
        }
        size_t j = std::distance(word.begin(), it);
        new_word.insert(new_word.end(), word.begin() + i, word.begin() + j);
        i = j;

        if (word[i] == first && i < word.size() - 1 && word[i + 1] == second) {
          new_word.push_back(first + second);
          i += 2;
        } else {
          new_word.push_back(word[i]);
          i += 1;
        }
      }
      word = new_word;
      if (word.size() == 1) {
        break;
      } else {
        pairs.clear();
        auto new_pairs = getPairs(word);
        pairs.insert(new_pairs.begin(), new_pairs.end());
      }
    }
  }

  void padTokens(std::vector<int32_t>& token) {
    if (token.size() < 77) {
      int zerosToAdd = 77 - token.size();
      for (int k = 0; k < zerosToAdd; ++k) {
        token.push_back(0);
      }
    } else {
      std::cerr << "Line statement is too long, shortened." << std::endl;
      token.resize(77);  // 或者根据需要处理
    }
  }
};
