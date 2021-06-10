#include "decode_tool.hpp"

#include <iostream>
#include <vector>

#define CODE_LENGTH 18
#define CHARS_NUM_TW 36
#define CHARS_NUM_CN 66

char const *CHAR_LIST_TW = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ-_";

// clang-format off
char const *CHAR_LIST_CN[CHARS_NUM_CN] = {
  "<Anhui>","<Shanghai>","<Tianjin>","<Chongqing>","<Hebei>",
  "<Shanxi>","<InnerMongolia>","<Liaoning>","<Jilin>","<Heilongjiang>",
  "<Jiangsu>","<Zhejiang>","<Beijing>","<Fujian>","<Jiangxi>",
  "<Shandong>","<Henan>","<Hubei>","<Hunan>","<Guangdong>",
  "<Guangxi>","<Hainan>","<Sichuan>","<Guizhou>","<Yunnan>",
  "<Tibet>","<Shaanxi>","<Gansu>","<Qinghai>","<Ningxia>",
  "<Xinjiang>",
  "0","1","2","3","4","5","6","7","8","9",
  "A","B","C","D","E","F","G","H","J","K",
  "L","M","N","P","Q","R","S","T","U","V",
  "W","X","Y","Z","_"};
// clang-format on

namespace LPR {

bool greedy_decode(float *y, std::string &id_number, LP_FORMAT format) {
  int chars_num;
  switch (format) {
    case TAIWAN:
      chars_num = CHARS_NUM_TW;
      break;
    case CHINA:
      chars_num = CHARS_NUM_CN;
      break;
    default:
      return false;
  }
  int code[CODE_LENGTH];
  for (int i = 0; i < CODE_LENGTH; i++) {
    float max_value = y[i];
    code[i] = 0;
    for (int j = 1; j < chars_num; j++) {
      if (y[j * CODE_LENGTH + i] > max_value) {
        max_value = y[j * CODE_LENGTH + i];
        code[i] = j;
      }
    }
  }

  switch (format) {
    case TAIWAN:
      id_number = decode_tw(code);
      break;
    case CHINA:
      id_number = decode_cn(code);
      break;
    default:
      return false;
  }
  return true;
}

std::string decode_tw(int *code) {
  std::vector<int> de_code;
  int previous = -1;
  for (int i = 0; i < CODE_LENGTH; i++) {
    if (code[i] != previous) {
      de_code.push_back(code[i]);
      previous = code[i];
    }
  }
  std::string id_number("");
  for (size_t i = 0; i < de_code.size(); i++) {
    if (de_code[i] != CHARS_NUM_TW - 1) {
      id_number.append(1, CHAR_LIST_TW[de_code[i]]);
    }
  }

  return id_number;
}

std::string decode_cn(int *code) {
  // printf("decode_cn\n");
  std::vector<int> de_code;
  int previous = -1;
  for (int i = 0; i < CODE_LENGTH; i++) {
    if (code[i] != previous) {
      de_code.push_back(code[i]);
      previous = code[i];
    }
  }
  std::string id_number("");
  for (size_t i = 0; i < de_code.size(); i++) {
    if (de_code[i] != CHARS_NUM_CN - 1) {
      id_number += CHAR_LIST_CN[de_code[i]];
    }
  }

  return id_number;
}

}  // namespace LPR