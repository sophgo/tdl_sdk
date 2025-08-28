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
  } else if (argc == 5) {
    std::string test_flag = argv[4];
    if (test_flag == "function") {
      cvitdl::unitest::CVI_TDLTestContext::getInstance().setTestFlag(
          cvitdl::unitest::TestFlag::FUNCTION);
    } else if (test_flag == "performance") {
      cvitdl::unitest::CVI_TDLTestContext::getInstance().setTestFlag(
          cvitdl::unitest::TestFlag::PERFORMANCE);
    } else if (test_flag == "generate_function_res") {
      cvitdl::unitest::CVI_TDLTestContext::getInstance().setTestFlag(
          cvitdl::unitest::TestFlag::GENERATE_FUNCTION_RES);
    } else if (test_flag == "generate_performance_res") {
      cvitdl::unitest::CVI_TDLTestContext::getInstance().setTestFlag(
          cvitdl::unitest::TestFlag::GENERATE_PERFORMANCE_RES);
    } else {
      printf("Invalid test flag: %s\n", test_flag.c_str());
      return -1;
    }
    printf("test_flag: %s,%d\n", test_flag.c_str(),
           cvitdl::unitest::CVI_TDLTestContext::getInstance().getTestFlag());
    testing::AddGlobalTestEnvironment(
        new cvitdl::unitest::CVI_TDLTestEnvironment(argv[1], argv[2], argv[3]));
    return RUN_ALL_TESTS();
  } else {
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
