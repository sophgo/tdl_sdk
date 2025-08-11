#ifndef TDL_TYPE_INTERNAL_EX_HPP
#define TDL_TYPE_INTERNAL_EX_HPP

#include "tdl_type_internal.hpp"
#if !defined(__CMODEL_CV181X__) && !defined(__CMODEL_CV184X__)
#include "network/api_poster/unified_api_client.hpp"
#endif

struct TDLContextEx {
  TDLContext core_context;
#if !defined(__CMODEL_CV181X__) && !defined(__CMODEL_CV184X__)
  std::unique_ptr<UnifiedApiClient> api_client;
#endif
};

#endif