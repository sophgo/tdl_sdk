#ifndef INCLUDE_COMMON_COMMON_H_
#define INCLUDE_COMMON_COMMON_H_

#include <log/Logger.hpp>
#include <map>
#include <vector>

#include <bmlib_runtime.h>



// Disable the copy and assignment operator for a class.
#define BM_DISABLE_COPY_AND_ASSIGN(classname)                                  \
private:                                                                       \
  classname(const classname &);                                                \
  classname &operator=(const classname &)

// A singleton class to hold common stuff, such as the handler that
// sdk_common is going to use for multi platform, etc.
class BMContext {
public:
  BMContext();
  ~BMContext();
  static BMContext &Get();

  bm_handle_t get_handle(int device_id);
  inline static bm_handle_t cnn_bm168x_handle(int device_id) {
    return Get().get_handle(device_id);
  }

  static void set_device_id(int device_id);

  std::map<int, bm_handle_t> device_handles_;


  pthread_mutex_t lock_ = PTHREAD_MUTEX_INITIALIZER;

  BM_DISABLE_COPY_AND_ASSIGN(BMContext);
};
#endif
