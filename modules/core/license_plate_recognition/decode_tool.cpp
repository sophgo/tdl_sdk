#include "decode_tool.hpp"

#include <iostream>
#include <vector>

char const *CHAR_LIST_TW = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ-_";

std::string greedy_decode(float *y) {
  int code[CODE_LENGTH];
  for (int i = 0; i < CODE_LENGTH; i++) {
    float max_value = y[i];
    code[i] = 0;
    for (int j = 1; j < CHARS_NUM_TW; j++) {
      if (y[j * CODE_LENGTH + i] > max_value) {
        max_value = y[j * CODE_LENGTH + i];
        code[i] = j;
      }
    }
  }
  std::string id_number = decode(code);
  return id_number;
}

std::string decode(int *code) {
  // printf("decode\n");
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