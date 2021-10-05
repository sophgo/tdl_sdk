#include "cviai_test.hpp"
#include "gtest.h"

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);

  if (argc != 4) {
    printf(
        "Usage: %s <model_dir>\n"
        "          <image_dir>\n"
        "          <regression_json>\n",
        argv[0]);
    return CVIAI_FAILURE;
  }

  testing::AddGlobalTestEnvironment(
      new cviai::unitest::CVIAITestEnvironment(argv[1], argv[2], argv[3]));
  return RUN_ALL_TESTS();
}
