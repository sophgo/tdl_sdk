#ifndef INCLUDE_COMMON_STATUS_H_
#define INCLUDE_COMMON_STATUS_H_

#include <cstdio>

/// status code enum.
typedef enum {
  BM_COMMON_SUCCESS = 0,
  BM_COMMON_NOT_SUPPORTED = 1,
  BM_COMMON_INVALID_DIVICE = 2,
  // related to arguments
  BM_COMMON_NULL_POINTER = 3,
  BM_COMMON_INVALID_ARGS = 4,
  BM_COMMON_OUT_OF_BOUND = 5,
  // failures of operations
  BM_COMMON_OP_NOT_PERMITTED = 7,
  BM_COMMON_ALLOC_FAILED = 8,
  BM_COMMON_EXECUTION_FAILED = 9,
  BM_COMMON_IO_ERROR = 10,
  // conficts with current status
  BM_COMMON_NOT_INITED = 11,
  BM_COMMON_ALREADY_INITED = 12,
  // others
  BM_COMMON_OTHER_ERROR = 255
} bmStatus_t;

inline const char* bmGetStatusName(const bmStatus_t s) {
  switch (s) {
    case BM_COMMON_SUCCESS:return "Success";
    case BM_COMMON_NOT_SUPPORTED:return "Not supported";
    case BM_COMMON_INVALID_DIVICE:return "Invalid device";
    case BM_COMMON_NULL_POINTER:return "Null pointer";
    case BM_COMMON_INVALID_ARGS:return "Invalid arguments";
    case BM_COMMON_OUT_OF_BOUND:return "Out of bound";
    case BM_COMMON_OP_NOT_PERMITTED:return "Operation is not permitted";
    case BM_COMMON_ALLOC_FAILED:return "Allocation failed";
    case BM_COMMON_EXECUTION_FAILED:return "Execution failed";
    case BM_COMMON_IO_ERROR:return "IO error";
    case BM_COMMON_NOT_INITED:return "Not initialized";
    case BM_COMMON_ALREADY_INITED:return "Already initialized";
    case BM_COMMON_OTHER_ERROR:return "Other internal error";
    default:return "Other undefined error";
  }
}

/// define check return status code macro.
#define BM_CHECK_STATUS(s)                                                  \
  {                                                                         \
    bmStatus_t st = s;                                                      \
    if (st != BM_COMMON_SUCCESS) {                                          \
      printf("Error: %s %d %s\n", __FILE__, __LINE__, bmGetStatusName(st)); \
      return st;                                                            \
    }                                                                       \
  }

#endif
