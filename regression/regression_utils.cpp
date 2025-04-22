#include "regression_utils.hpp"



namespace cvitdl {
namespace unitest {

  std::string gen_model_suffix(){

#if defined(__CV181X__) || defined(__CMODEL_CV181X__)
  return std::string("_cv181x.cvimodel");

#elif defined(__CV186X__)
  return std::string("_cv186x.bmodel");

#elif defined(__BM1684X__)
  return std::string("_bm1684x.bmodel");  

#elif defined(__BM168X__)
  return std::string("_bm1688.bmodel");  

#else
  printf("Unrecognized platform !\n");
  return std::string("");

#endif
  }


  std::string gen_platform(){

#if defined(__CV181X__) || defined(__CMODEL_CV181X__)
  return std::string("CV181X");

#elif defined(__CV186X__)
  return std::string("CV186X");  

#elif defined(__BM1684X__)
  return std::string("BM1684X");  

#elif defined(__BM168X__)
  return std::string("BM1688");  

#else
  printf("Unrecognized platform !\n");
  return std::string("");

#endif
  }


}  // namespace unitest
}  // namespace cvitdl
