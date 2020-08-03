#include "coco_utils.hpp"
#include <memory.h>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>

using namespace std;

namespace cviai {
namespace coco_utils {

struct ClassIdMapSingleton {
 public:
  static ClassIdMapSingleton *get_instance() {
    ClassIdMapSingleton *tmp = instance.load(memory_order_relaxed);
    atomic_thread_fence(memory_order_acquire);
    if (tmp == nullptr) {
      lock_guard<mutex> lock(instanceMutex);
      tmp = instance.load(memory_order_relaxed);
      if (tmp == nullptr) {
        tmp = new ClassIdMapSingleton;
        atomic_thread_fence(memory_order_release);
        instance.store(tmp, memory_order_relaxed);
      }
    }
    return tmp;
  }

  int map_class_80(size_t classid_90) { return m_lut_90_to_80[classid_90]; }

 private:
  ClassIdMapSingleton() {
    m_lut_90_to_80 = new int[class_names_90.size()];
    memset(m_lut_90_to_80, -1, class_names_90.size() * sizeof(int));

    size_t count = 0;
    for (size_t index_90 = 0; index_90 < class_names_90.size(); index_90++) {
      auto iter = find(class_names_80.begin(), class_names_80.end(), class_names_90[index_90]);
      if (iter != class_names_80.end()) {
        int index_80 = distance(class_names_80.begin(), iter);
        m_lut_90_to_80[index_90] = index_80;
        count++;
      }
    }
  }

  ~ClassIdMapSingleton() { delete[] m_lut_90_to_80; }

  ClassIdMapSingleton(const ClassIdMapSingleton &) = delete;
  ClassIdMapSingleton &operator=(const ClassIdMapSingleton &) = delete;

  int *m_lut_90_to_80;
  static std::atomic<ClassIdMapSingleton *> instance;
  static std::mutex instanceMutex;
};

std::mutex ClassIdMapSingleton::instanceMutex;
std::atomic<ClassIdMapSingleton *> ClassIdMapSingleton::instance;

int map_90_class_id_to_80(int class_id) {
  ClassIdMapSingleton *pinstance = ClassIdMapSingleton::get_instance();
  return pinstance->map_class_80(class_id);
}
}  // namespace coco_utils
}  // namespace cviai