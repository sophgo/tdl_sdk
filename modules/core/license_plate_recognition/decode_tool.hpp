#ifndef _LP_DECODE_TOOL_H_
#define _LP_DECODE_TOOL_H_
#include <string>

#define CODE_LENGTH 18
#define CHARS_NUM_TW 36

std::string greedy_decode(float *y);

std::string decode(int *code);

#endif /* _LP_DECODE_TOOL_H_ */