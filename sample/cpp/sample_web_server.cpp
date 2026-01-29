#include <fstream>  // 添加 fstream 头文件
#include <iostream>
#include "components/media_analysis/httplib.h"

// 修改main函数签名以支持命令行参数
int main(int argc, char* argv[]) {
  using namespace httplib;

  // 检查是否提供了dist路径参数
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <dist_path>" << std::endl;
    return 1;
  }
  std::string dist_path = argv[1];  // 存储命令行传入的dist路径

  Server svr;

  // 1. 设置静态文件目录，使用命令行传入的路径
  auto ret = svr.set_mount_point("/", dist_path.c_str());

  if (!ret) {
    std::cerr << "错误：无法找到 dist 目录，请检查路径！" << std::endl;
    return 1;
  }

  // 2. 特别处理：React路由404重定向，使用动态dist路径
  svr.set_error_handler([dist_path](const Request& /*req*/,
                                    Response& res) {  // 捕获dist_path
    if (res.status == 404) {
      // 使用命令行传入的路径拼接index.html
      std::string index_path = dist_path + "/index.html";
      std::ifstream ifs(index_path);
      if (ifs) {
        std::string content((std::istreambuf_iterator<char>(ifs)),
                            (std::istreambuf_iterator<char>()));
        res.set_content(content, "text/html");
        res.status = 200;
      } else {
        std::cerr << "Error: index.html not found at " << index_path
                  << std::endl;
        res.status = 404;
        res.set_content("Error: index.html not found. Please check dist path.",
                        "text/plain");
      }
    }
  });

  std::cout << "服务器启动在 http://0.0.0.0:8080" << std::endl;

  // 3. 监听所有网卡上的8080端口
  svr.listen("0.0.0.0", 8080);

  return 0;
}
