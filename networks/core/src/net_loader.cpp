
#include "net_loader.hpp"
#include "bmodel_runner.hpp"
#include "model_runner.hpp"

namespace qnn {

using std::shared_ptr;
using std::string;

shared_ptr<ModelRunner> NetLoader::Load(const string &model) {
    auto res = path_to_runner.find(model);
    if (res == path_to_runner.end()) {
        res = path_to_runner.insert({model, std::make_shared<BModelRunner>(ctxt_handle, model)})
                  .first;
    }
    return res->second;
}

}  // namespace qnn
