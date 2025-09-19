#include <gtest/gtest.h>
#include "cvi_tdl_test.hpp"

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);

  if (argc == 2) {
    testing::InitGoogleTest(&argc, argv);
    setenv("MATCHER_TYPE", argv[1], 1);
    return RUN_ALL_TESTS();
  } else if (argc == 3) {
    testing::AddGlobalTestEnvironment(
        new cvitdl::unitest::CVI_TDLTestEnvironment(argv[1], argv[2], ""));
    return RUN_ALL_TESTS();
  } else if (argc == 5) {
    std::string test_flag = argv[4];
    if (!cvitdl::unitest::CVI_TDLTestContext::getInstance().setTestFlag(
            test_flag)) {
      printf("Invalid test flag: %s\n", test_flag.c_str());
      return -1;
    }

    testing::AddGlobalTestEnvironment(
        new cvitdl::unitest::CVI_TDLTestEnvironment(argv[1], argv[2], argv[3]));
    return RUN_ALL_TESTS();
  } else {
    printf("Usage: %s <model_dir> <regress_asset_dir>\n", argv[0]);
    printf("Usage: %s <model_dir> <regress_asset_dir> <json_file_name>\n",
           argv[0]);
    printf(
        "Usage: %s <model_dir> <regress_asset_dir> <json_file_name> "
        "<test_flag>\n",
        argv[0]);
    printf(
        "test_flag: "
        "function,performance,generate_function_res,generate_performance_"
        "res\n");
    return -1;
  }
}
