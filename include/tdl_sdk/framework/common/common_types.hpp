#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <cstdint>
#include <string>
#include <vector>

enum class InferencePlatform {
  UNKOWN = 0,
  CVITEK = 1,
  CV186X = 2,
  BM168X = 3,
  CMODEL = 4,
  AUTOMATIC = 5
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

enum class ImageImplType {
  VPSS_FRAME = 0,
  OPENCV_FRAME,
  FFMPEG_FRAME,
  BMCV_FRAME,
  TENSOR_FRAME,  // input tensor frame
  RAW_FRAME,     // raw frame,just a block of continuous memory
  UNKOWN
};

struct MemoryBlock {
  uint64_t physicalAddress;  // 内存块的物理地址（仅用于 SoC 场景）
  void *virtualAddress;      // 内存块的虚拟地址
  void *handle;
  uint64_t size;  // 内存块的大小（字节数）
  uint32_t id;
  bool own_memory;
  MemoryBlock() {
    physicalAddress = 0;
    virtualAddress = nullptr;
    handle = nullptr;
    size = 0;
    id = UINT32_MAX;
    own_memory = false;
  }
};

struct PreprocessParams {
  ImageFormat dstImageFormat;
  TDLDataType dstPixDataType;
  int dstWidth;
  int dstHeight;
  int cropX;
  int cropY;
  int cropWidth;
  int cropHeight;
  float mean[3];
  float scale[3];  // Y=X*scale+mean
  bool keepAspectRatio;
};

enum class MemoryType {
  HOST_MEMORY = 0,
  ASIC_DEVICE_MEMORY = 1,
};

struct NetParam {
  InferencePlatform platform;
  int device_id = 0;

  // bool share_output_mem = false;  // do not allocate output tensor memory,
  // share with other memory
  std::string model_file_path;
  std::string net_name;  // Specifies the network name in case of multiple
                         // networks in a model
  std::vector<std::string>
      input_names;  // Leave empty to read input nodes from model file
  std::vector<std::string> output_names;
  PreprocessParams
      pre_params;  // TODO(fuquan.ke) to support multiple preprocess params
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

#endif  // COMMON_TYPES_H