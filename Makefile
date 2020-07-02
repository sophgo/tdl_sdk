#
PWD := $(shell pwd)
#
ifeq ($(SDK_VER),"32bit")
CC := arm-linux-gnueabihf-gcc
CXX := arm-linux-gnueabihf-g++
else
CC := $(TOOLCHAIN_PATH)/aarch64-linux-gnu-gcc
CXX := $(TOOLCHAIN_PATH)/aarch64-linux-gnu-g++
endif

TARGET := libcvialgo.so

LDFLAGS = -shared -fPIC -o


SRC_CPP := $(wildcard ./modules/*.cpp)
SRC_CPP += $(wildcard ./modules/cvi_face_attribute/*.cpp)
SRC_CPP += $(wildcard ./modules/cvi_feature_matching/*.cpp)
SRC_CPP += $(wildcard ./modules/cvi_liveness/*.cpp)
SRC_CPP += $(wildcard ./modules/cvi_retina_face/*.cpp)
SRC_CPP += $(wildcard ./modules/cvi_utils/*.cpp)
SRC_CPP += $(wildcard ./modules/cvi_yolove3/*.cpp)
SRC_CPP += $(wildcard ./modules/cvi_thermal_face_detect/*.cpp)
SRC_C += $(shell find ./modules -name *.c)
#
OBJS := $(SRC_C:%.c=%.o)
OBJS += $(SRC_CPP:%.cpp=%.o)
#
MLIR_INC ?= $(SDK_INSTALL_PATH)/tpu/cvitek_tpu_sdk/include
#MLIR_LIB ?=  -L$(SDK_INSTALL_PATH)/tpu/cvitek_tpu_sdk/lib  -lcviruntime -lcvikernel -lcnpy -lcvimath
#MLIR_LIB ?=  -lcviruntime -lcvikernel -lcnpy 
#MLIR_LIB += -L$(SDK_PATH)/oss/build_zlib -lz
#MLIR_LIB +=  -L$(SDK_INSTALL_PATH)/system/lib -lcvimath
MLIR_LIB ?= -L/home/terry/work/easy_build/prebuilt/soc_bm1880v2_asic/system/lib -lcviruntime -lcvikernel -lcnpy -lz -lcvimath

SAMPLE_INC ?= $(SDK_PATH)/middleware/sample/common

# PREBUILT_DIR = $(SDK_INSTALL_PATH)/rootfs/lib

OPENCV_LIB = -L/home/terry/work/cvi_pipeline/cvitek_tpu_sdk/opencv/lib -lopencv_imgcodecs -lopencv_core -lopencv_imgproc
OPENCV_INC = /home/terry/work/cvi_pipeline/cvitek_tpu_sdk/opencv/include

# PREBUILT_INC = $(SDK_PATH)/ramdisk/prebuild/include

#Please set your path. 
#LIB_DIR += -L$(PREBUILT_DIR)/libprotobuf/lib
#LIB_DIR += -L$(SDK_PATH)/middleware/modules/venc/vc_lib/lib
# LIB_DIR += -L$(PREBUILT_DIR) 
LIB_DIR += -L$(SAMPLE_INC)
#LIB_DIR += -L$(PWD)/customize/demo_src/camera/fam600/lib
LIB_DIR += -L$(SDK_PATH)/middleware/lib
LIB_DIR += -L$(PWD)/../app_sample/lib
LIBS += $(MLIR_LIB) $(OPENCV_LIB)
LIBS += -L/home/terry/work/access-guard/function-lib/db/src/ -lconfig_utils

INCLUDES += -I$(SDK_PATH)/middleware/include
INCLUDES += -I$(OPENCV_INC)
INCLUDES += -I$(PWD)/modules/include
INCLUDES += -I$(MLIR_INC)
INCLUDES += -I$(PWD)/../function_lib/db/include
INCLUDES += -I$(PWD)/../function_lib/db/sqlite3/include
#

ifeq ($(CUSTOMER), giwei)
	DEFS = -DACCESS_GUARD 
	CFLAGS += $(DEFS)
endif


#
INCLUDES += -I/home/terry/work/easy_build/prebuilt/soc_bm1880v2_asic/system/include
INCLUDES += -I$(PWD)/modules/

ifeq ($(CUSTOMER), giwei)
	DEFS = -DACCESS_GUARD 
	CFLAGS += $(DEFS)
endif

$(TARGET): $(OBJS)
	$(CXX)  $(LDFLAGS) $(TARGET) $(OBJS)  $(LIB_DIR) $(LIBS)

%.o: %.c
	$(CC) -O3 -fPIC -c $< -o $@ $(INCLUDES) $(CFLAGS)
%.o: %.cpp
	$(CXX) -O3 -fPIC -c $< -o $@ $(INCLUDES) $(CFLAGS)

.PHONY: clean  
clean:
	rm -rf $(OBJS) $(TARGET)
