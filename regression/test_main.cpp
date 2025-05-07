#include "cvi_tdl_test.hpp"
#include "gtest.h"

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);

  if (argc == 2) {
    testing::InitGoogleTest(&argc, argv);
    setenv("MATCHER_TYPE", argv[1], 1);
    return RUN_ALL_TESTS();
  } else if (argc == 4) {
    testing::AddGlobalTestEnvironment(
        new cvitdl::unitest::CVI_TDLTestEnvironment(argv[1], argv[2], argv[3]));
    return RUN_ALL_TESTS();
  } else {
    printf("Usage: %s <matcher_type> --gtest_filter=MatcherTestSuite.*\n",
           argv[0]);
    printf(
        "Usage: %s <model_dir>\n"
        "          <image_dir>\n"
        "          <regression_json>\n",
        argv[0]);
    return -1;
  }
}
