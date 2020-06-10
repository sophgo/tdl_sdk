// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
/**
 * @file function_tracer.h
 * @ingroup NETWORKS_DEBUG
 */
#pragma once
#include <assert.h>
#include <cstring>

// #include "config.h"
#include "log_common.h"
#include "tracer.h"

#if ENABLE_FUNCTION_TRACE && ENABLE_FUNCTION_SYSTRACE
#    error "Both ENABLE_FUNCTION_TRACE & ENABLE_FUNCTION_SYSTRACE are ON"
#elif ENABLE_FUNCTION_TRACE
#    define BITMAIN_FUNCTION_TRACE(char_arr) FunctionTracer t(char_arr)
#elif ENABLE_FUNCTION_SYSTRACE
#    define BITMAIN_FUNCTION_TRACE(char_arr) ScopedTrace t(char_arr)
#else
#    define BITMAIN_FUNCTION_TRACE(char_arr)
#endif

class FunctionTracer {
    public:
    explicit FunctionTracer(const char *c) {
        assert(std::strlen(c) > 0);
        m_func_name = new char[std::strlen(c) + 1];
        std::strcpy(m_func_name, c);
        char arr[] = " start";
        char *msg = new char[std::strlen(m_func_name) + std::strlen(arr) + 1];
        std::strcpy(msg, m_func_name);
        std::strcat(msg, arr);
        LOGD << msg;
        delete[] msg;
    }
    ~FunctionTracer() {
        char arr[] = " end";
        char *msg = new char[std::strlen(m_func_name) + std::strlen(arr) + 1];
        std::strcpy(msg, m_func_name);
        std::strcat(msg, arr);
        LOGD << msg;
        delete[] msg;
        delete[] m_func_name;
    }

    private:
    char *m_func_name = nullptr;
    // bool m_print_flag = 0;
};

/* TODO: Shared memory
   int my_shmid = shmget(key,size,shmflgs);
   void* address_of_my_shm1 = shat(my_shmid,0,shmflags);
   Object* optr = static_cast<Object*>(address_of_my_shm1); */
