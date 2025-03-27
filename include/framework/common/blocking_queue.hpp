#ifndef __NPU_BLOCKING_QUEUE_HPP__
#define __NPU_BLOCKING_QUEUE_HPP__

// #include "profiler.hpp"
#include <errno.h>
#include <pthread.h>
#include <string.h>
#include <sys/time.h>
#include <iostream>
#include <queue>
#include <string>
// #define BLOCKING_QUEUE_PERF
template <typename T>
class BlockingVal {
 public:
  BlockingVal() {
    lock_ = PTHREAD_MUTEX_INITIALIZER;
    memset(&val_, 0, sizeof(val_));
  }
  ~BlockingVal() {}
  T val_;
  pthread_mutex_t lock_;
  void set(T val) {
    pthread_mutex_lock(&lock_);
    val_ = val;
    pthread_mutex_unlock(&lock_);
  }
  T add(T val) {
    T tmp;
    pthread_mutex_lock(&lock_);
    // int *p_dst = static_cast<int*>(&val_);
    //*p_dst = *p_dst + static_cast<int>(val);
    val_ += val;
    tmp = val_;
    pthread_mutex_unlock(&lock_);
    return tmp;
  }
  T get() {
    T val;
    pthread_mutex_lock(&lock_);
    val = val_;
    pthread_mutex_unlock(&lock_);
    return val;
  }
  T get_unsafe() { return val_; }
};

template <typename T>
class BlockingQueue {
 public:
  BlockingQueue(const std::string &name = "", int type = 0);
  ~BlockingQueue();

  void stop();
  int push(T data);
  int push(const std::vector<T> &data);
  void drop(int num);
  T pop(int wait_ms = 0, bool *is_timeout = nullptr);
  T pop_at(int pop_idx);
  void lock() { pthread_mutex_lock(&m_qmtx); }
  void unlock() { pthread_mutex_unlock(&m_qmtx); }
  std::vector<T> &get_queue() { return m_vec; }
  void set_name(std::string strname) { m_name = strname; }
  void signal() { pthread_cond_signal(&m_condv); }
  size_t size();
  size_t sizeUnsafe();

 private:
  bool m_stop;
  //   Timer m_timer;
  std::string m_name;
  std::vector<T> m_vec;
  std::queue<T> m_queue;
  pthread_mutex_t m_qmtx;
  pthread_cond_t m_condv;
  int m_type;  // 0:queue,1:vector
};

// #include "blocking_queue.hpp"

template <typename T>
BlockingQueue<T>::BlockingQueue(const std::string &name, int type) : m_stop(false) {
  m_name = name;
  m_type = type;
  pthread_mutex_init(&m_qmtx, NULL);
  pthread_cond_init(&m_condv, NULL);
}

template <typename T>
BlockingQueue<T>::~BlockingQueue() {
  pthread_mutex_lock(&m_qmtx);
  //  std::queue<T> empty;
  //  m_queue.swap(empty);
  std::cout << "destroy " << m_name << ",size:" << m_queue.size() + m_vec.size() << std::endl;
  m_vec.clear();
  std::queue<T> empty;
  m_queue.swap(empty);
  pthread_mutex_unlock(&m_qmtx);
}

template <typename T>
void BlockingQueue<T>::stop() {
  m_stop = true;
  pthread_cond_broadcast(&m_condv);
  std::cout << "stop blocking queue :" << m_name << std::endl;
}

template <typename T>
int BlockingQueue<T>::push(T data) {
  int num = 0;
  // std::cout<<m_name<<" to push\n";
  pthread_mutex_lock(&m_qmtx);
  if (m_type == 0) {
    m_queue.push(std::move(data));
    num = m_queue.size();
  } else {
    m_vec.emplace_back(std::move(data));
    num = m_vec.size();
  }

  pthread_mutex_unlock(&m_qmtx);
  pthread_cond_signal(&m_condv);
  // std::cout<<m_name<<" after push,num:"<<num<<std::endl;
  return num;
}
template <typename T>
int BlockingQueue<T>::push(const std::vector<T> &datas) {
  int num = 0;
  pthread_mutex_lock(&m_qmtx);
  if (m_type == 0) {
    for (int i = 0; i < datas.size(); i++) m_queue.push(std::move(datas[i]));
    num = m_queue.size();
  } else {
    for (int i = 0; i < datas.size(); i++) m_vec.emplace_back(std::move(datas[i]));
    num = m_vec.size();
  }

  pthread_mutex_unlock(&m_qmtx);
  pthread_cond_signal(&m_condv);
  return num;
}
template <typename T>
T BlockingQueue<T>::pop(int wait_ms, bool *is_timeout) {
  T ret;
  bool timeout = false;
  // std::cout<<"watits:"<<wait_ms<<std::endl;
  if (m_stop) {
    return ret;
  }

  struct timespec to;
  struct timeval now;
  gettimeofday(&now, NULL);
  // double ms0 = now.tv_sec * 1000 + now.tv_usec / 1000.0;
  // std::cout<<m_name<<",pop:"<<now.tv_usec/1000.0;
  if (wait_ms == 0) {
    to.tv_sec = now.tv_sec + 9999999;
    to.tv_nsec = now.tv_usec * 1000UL;
  } else {
    int nsec = now.tv_usec * 1000 + (wait_ms % 1000) * 1000000;
    to.tv_sec = now.tv_sec + nsec / 1000000000 + wait_ms / 1000;
    to.tv_nsec = nsec % 1000000000;  //(now.tv_usec + wait_ms * 1000UL) * 1000UL;
  }
  // std::cout<<m_name<<" BlockingQueue
  // topop:"<<wait_ms<<",cursize:"<<sizeUnsafe()<<",datasize:"<<sizeof(T)<<std::endl;
  pthread_mutex_lock(&m_qmtx);
  while ((m_type ? m_vec.empty() : m_queue.empty()) && !m_stop) {
#ifdef BLOCKING_QUEUE_PERF
    m_timer.tic();
#endif

    // pthread_timestruc_t to;
    int err = pthread_cond_timedwait(&m_condv, &m_qmtx, &to);
    if (err == ETIMEDOUT || m_stop) {
      timeout = true;
      break;
    }
#ifdef BLOCKING_QUEUE_PERF
    m_timer.toc();
    if (m_timer.total_time_ > 1) {
      m_timer.summary();
    }
#endif
  }
  if (!timeout && !m_stop) {
    if (m_type == 0) {
      // ret = m_queue.front();
      ret = std::move(m_queue.front());
      m_queue.pop();
    } else {
      ret = std::move(m_vec[0]);
      m_vec.erase(m_vec.begin());
    }
  }
  pthread_mutex_unlock(&m_qmtx);

  if (is_timeout) {
    *is_timeout = timeout;
  }
  // std::cout<<"BlockingQueue to return,left:"<<sizeUnsafe()<<std::endl;
  return ret;
}

template <typename T>
T BlockingQueue<T>::pop_at(int pop_idx) {
  // lock should be called first
  assert(pop_idx >= 0 && pop_idx < m_vec.size());
  T ret = std::move(m_vec[pop_idx]);
  m_vec.erase(m_vec.begin() + pop_idx);
  return ret;
}

template <typename T>
size_t BlockingQueue<T>::size() {
  size_t queue_size;

  pthread_mutex_lock(&m_qmtx);
  if (m_type == 0)
    queue_size = m_queue.size();
  else
    queue_size = m_vec.size();
  pthread_mutex_unlock(&m_qmtx);

  return queue_size;
}
template <typename T>
size_t BlockingQueue<T>::sizeUnsafe() {
  size_t queue_size;

  if (m_type == 0)
    queue_size = m_queue.size();
  else
    queue_size = m_vec.size();
  return queue_size;
}

template <typename T>
void BlockingQueue<T>::drop(int num) {
  // lock should be called first
  int cur_size = sizeUnsafe();
  if (cur_size < num) return;
  int queue_size;
  pthread_mutex_lock(&m_qmtx);
  if (m_type == 0) {
    queue_size = m_queue.size();
    if (num > queue_size) num = queue_size;
    for (int i = 0; i < num; i++) {
      m_queue.pop();
    }
  } else {
    queue_size = m_vec.size();
    if (num > queue_size) num = queue_size;
    m_vec.erase(m_vec.begin(), m_vec.begin() + num);
  }
  pthread_mutex_unlock(&m_qmtx);
}
#endif
