# TDLSDK Regression

## Google Test简介

- Google Test（gtest）是一个由Google提供的C++单元测试框架, 提供了丰富的断言类型和辅助函数，使得C++测试用例变得更简洁和直观。
- GoogleTest提供了一系列的断言宏来检查条件是否满足。如果断言失败，测试用例被认为失败。断言分为两大类：ASSERT_*和 EXPECT_*。
- ASSERT_* 在断言失败时会产生一个致命错误，并终止当前函数的执行。主要有以下类型：

| ASSERT类型                              | 作用                                     |
| --------------------------------------- | ---------------------------------------- |
| ASSERT_EQ (val1, val2)                 | 断言两个值相等（整数、指针等）           |
| ASSERT_LT (val1, val2)                 | 断言val1小于val2                         |
| ASSERT_LE (val1, val2)                 | 断言val1小于val2                         |
| ASSERT_GT (val1, val2)                 | 断言val1大于val2                         |
| ASSERT_GE (val1, val2)                 | 断言val1大于等于val2                     |
| ASSERT _FLOAT_EQ (val1, val2)          | 断言两个float 类型的值相等               |
| ASSERT_NEAR(val1, val2，abs_error)     | 断言val1, val2差值的绝对值小于abs_error  |
| ASSERT_TRUE(condition)                 | 断言condition为true                      |
| ASSERT_STREQ (s1, s2)                  | 断言两个C字符串相等                      |
| ASSERT_NO_FATAL_FAILURE (statement)    | 断言代码运行时没有致命错误               |
| ASSERT_PRED2 (pred, val1, val2)        | 断言自定义的比较方式                     |

- EXPECT_* 在断言失败时会产生一个非致命错误，测试将不会立即终止，而是会记录失败信息并继续执行后续的测试代码，使用方式与ASSERT类似。

## 回归测试执行方法

```shell
./daily_regression.sh -m models \                      # 模型所在文件夹的上级目录
                      -d aisdk_daily_regression \      #aisdk_daily_regression仓库目录
                      -a aisdk_daily_regression/json  #json目录，在aisdk_daily_regression下面

```
