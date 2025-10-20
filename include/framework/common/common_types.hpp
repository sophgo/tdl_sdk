#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

enum class InferencePlatform {
  UNKOWN = 0,
  CVITEK = 1,
  CV186X = 2,
  CV184X = 3,
  BM168X = 4,
  CMODEL_CV181X = 5,
  CMODEL_CV184X = 6,
  AUTOMATIC = 7
};

enum class TDLDataType {
  INT8 = 0,
  UINT8 = 1,
  INT16 = 2,
  UINT16 = 3,
  INT32 = 4,
  UINT32 = 5,
  BF16 = 6,
  FP16 = 7,
  FP32 = 8,
  UNKOWN
};

enum class ImageFormat {
  GRAY = 0,
  RGB_PLANAR,
  RGB_PACKED,
  BGR_PLANAR,
  BGR_PACKED,
  YUV420SP_UV,  // NV12,semi-planar,one Y plane,one interleaved UV plane,size =
                // width * height * 1.5
  YUV420SP_VU,  // NV21,semi-planar,one Y plane,one interleaved VU plane,size =
                // width * height * 1.5
  YUV420P_UV,   // I420,planar,one Y plane(w*h),one U plane(w/2*h/2),one V
                // plane(w/2*h/2),size = width * height * 1.5
  YUV420P_VU,   // YV12,size = width * height * 1.5
  YUV422P_UV,   // I422_16,size = width * height * 2
  YUV422P_VU,   // YV12_16,size = width * height * 2
  YUV422SP_UV,  // NV16,size = width * height * 2
  YUV422SP_VU,  // NV61,size = width * height * 2

  UNKOWN
};

enum class ImageType {
  VPSS_FRAME = 0,
  OPENCV_FRAME,
  FFMPEG_FRAME,
  BMCV_FRAME,
  TENSOR_FRAME,  // input tensor frame
  RAW_FRAME,     // raw frame,just a block of continuous memory
  UNKOWN
};

enum class MemoryType {
  UNKOWN = 0,
  HOST_MEMORY = 1,
  SOC_DEVICE_MEMORY = 2,   // could have virtual address
  PCIE_DEVICE_MEMORY = 3,  // only device address
};

struct MemoryBlock {
  uint64_t physicalAddress;  // 内存块的物理地址（仅用于 SoC 场景）
  void *virtualAddress;      // 内存块的虚拟地址
  // void *handle;

  uint64_t size;  // 内存块的大小（字节数）
  uint32_t id;
  bool own_memory;
  MemoryType memory_type;
  MemoryBlock() {
    physicalAddress = 0;
    virtualAddress = nullptr;
    // handle = nullptr;
    size = 0;
    id = UINT32_MAX;
    own_memory = false;
    memory_type = MemoryType::UNKOWN;
  }
};

struct PreprocessParams {
  ImageFormat dst_image_format;
  TDLDataType dst_pixdata_type;
  int dst_width;
  int dst_height;
  int crop_x;
  int crop_y;
  int crop_width;
  int crop_height;
  float mean[3];
  float scale[3];  // Y=X*scale-mean
  bool keep_aspect_ratio;
};

struct ModelConfig {
  std::string
      net_name;  // specify the network name in case of multiple networks in a
                 // model,if only one network inside the model,could leave empty
  std::string file_name;
  std::string comment;
  std::vector<float> mean;
  std::vector<float> std;
  std::string rgb_order;  // rgb,bgr,gray,if not specified,leave emp

  // the model derived from BaseModel use these information to initialize
  std::vector<std::string> types;  // specify the category types,if not
                                   // specified,leave empty
  std::map<std::string, std::string>
      custom_config_str;  // could use additional info to parse input or output
                          // node,if not necessary,leave empty
  std::map<std::string, int>
      custom_config_i;  // custom int type config,if not necessary,leave empty
  std::map<std::string, float>
      custom_config_f;  // custom float type config,if not necessary,leave empty
};

struct NetParam {
  InferencePlatform platform;
  int device_id = 0;

  std::string model_file_path;
  uint8_t *model_buffer;  // if model_file_path is not set,use model_buffer to
                          // load model
  uint32_t model_buffer_size = 0;

  std::vector<uint64_t> runtime_mem_addrs;  // the size should be 0 or 5
  std::vector<uint32_t> runtime_mem_sizes;  // the size should be equal to the
                                            // size of runtime_mem_addrs

  ModelConfig model_config;
};

struct TensorInfo {
  void *tensor_handle;
  uint8_t *sys_mem;
  uint64_t phy_addr;
  std::vector<int> shape;
  TDLDataType data_type;
  // Tensor size = (number of tensor elements) * sizeof(data_type type))
  uint32_t tensor_size;
  // number of tensor elements
  uint32_t tensor_elem;
  float qscale;
  int zero_point;
};

struct LLMInferParam {
  int max_new_tokens;
  float top_p;
  float temperature;
  float repetition_penalty;
  int repetition_last_n;
  std::string generation_mode;
  std::string prompt_mode;
};

#endif  // COMMON_TYPES_H