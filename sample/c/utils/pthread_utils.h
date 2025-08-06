#ifndef PTHREAD_UTILS_H_
#define PTHREAD_UTILS_H_

#include <pthread.h>

#define MUTEXAUTOLOCK_INIT(mutex) \
  pthread_mutex_t AUTOLOCK_##mutex = PTHREAD_MUTEX_INITIALIZER;

#define MutexAutoLock(mutex, lock)                             \
  __attribute__((cleanup(AutoUnLock))) pthread_mutex_t *lock = \
      &AUTOLOCK_##mutex;                                       \
  pthread_mutex_lock(lock);

__attribute__((always_inline)) inline void AutoUnLock(void *mutex) {
  pthread_mutex_unlock(*(pthread_mutex_t **)mutex);
}

#endif
