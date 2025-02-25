#include "core/cvi_tdl_types_mem.h"
#include "core/cvtdl_core_types.h"
#include "image/base_image.hpp"
#include "models/tdl_model_factory.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <model_dir> <image_path>\n", argv[0]);
        return -1;
    }

    std::string model_dir = argv[1];
    std::string image_path = argv[2];

    auto image = ImageFactory::readImage(image_path);
    if (!image) {
        printf("Failed to load images\n");
        return -1;
    }

    TDLModelFactory model_factory(model_dir);
    
    std::shared_ptr<BaseModel> model_cls = 
        model_factory.getModel(TDL_MODEL_TYPE_FACE_ANTI_SPOOF_CLASSIFICATION);

    if (!model_cls) {
        printf("Failed to load classification model\n");
        return -1;
    }


    std::vector<std::shared_ptr<BaseImage>> input_images = {image};
    
    std::vector<void*> out_cls;
    model_cls->inference(input_images, out_cls);
    for (size_t i = 0; i < input_images.size(); i++) {
        cvtdl_class_meta_t* cls_meta = static_cast<cvtdl_class_meta_t*>(out_cls[i]);
        printf("pred_label: %d\n", cls_meta->cls[0]);
    }

    model_factory.releaseOutput(TDL_MODEL_TYPE_FACE_ANTI_SPOOF_CLASSIFICATION, out_cls);

    return 0;
}