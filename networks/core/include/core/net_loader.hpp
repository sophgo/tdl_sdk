/**
 * @file net_loader.hpp
 * @ingroup NETWORKS_CORE
 */
#pragma once

#include "bmtap2.h"
#include "model_runner.hpp"
#include <memory>
#include <string>
#include <unordered_map>

namespace qnn {

// This looks like model manager?
/**
 * @brief The singleton model loader
 *
 */
class NetLoader {
    public:
    /**
     * @brief Singleton getter function
     *
     * @return NetLoader& return singleton object
     */
    static NetLoader& Get() {
        static NetLoader loader;
        return loader;
    }

    /**
     * @brief Load and create shared pointer from bmodel file.
     *
     * @param model Model path
     * @return std::shared_ptr<ModelRunner> ModelRunner share pointer
     */
    std::shared_ptr<ModelRunner> Load(const std::string& model);

    /**
     * @brief Get the bm context.
     *
     * @return bmctx_t The bm context
     */
    bmctx_t GetBmCtxt() { return ctxt_handle; }

    private:
    /**
     * @brief Construct a new NetLoader object.
     *
     */
    NetLoader() { bm_init(0, &ctxt_handle); }

    /**
     * @brief Destroy the NetLoader object.
     *
     */
    ~NetLoader() {
        path_to_runner.clear();
        bm_exit(ctxt_handle);
    }

    bmctx_t ctxt_handle; /**< The bm context */
    std::unordered_map<std::string, std::shared_ptr<ModelRunner>>
        path_to_runner; /**< The ModelRunner unordered map container with path info and instance */
};

}  // namespace qnn
