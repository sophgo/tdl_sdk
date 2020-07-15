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

SRC_CPP := $(wildcard ./modules/cv183x_facelib.cpp)
SRC_CPP += $(wildcard ./modules/cvi_face_attribute/*.cpp)
SRC_CPP += $(wildcard ./modules/cvi_feature_matching/*.cpp)
SRC_CPP += $(wildcard ./modules/cvi_liveness/*.cpp)
SRC_CPP += $(wildcard ./modules/cvi_retina_face/*.cpp)
SRC_CPP += $(wildcard ./modules/cvi_utils/*.cpp)
SRC_CPP += $(wildcard ./modules/cvi_yolove3/*.cpp)
SRC_CPP += $(wildcard ./modules/cvi_thermal_face_detect/*.cpp)
SRC_CPP += $(wildcard ./modules/cvi_face_quality/*.cpp)
SRC_C += $(shell find ./modules -name *.c)
OBJS := $(SRC_C:%.c=%.o)
OBJS += $(SRC_CPP:%.cpp=%.o)

SDK_INC = $(SDK_PATH)/include
SDK_LIB = -L$(SDK_PATH)/lib -lcviruntime -lcvikernel -lcnpy -lz -lcvimath
OPENCV_LIB = -L$(SDK_PATH)/opencv/lib -lopencv_imgcodecs -lopencv_core -lopencv_imgproc
OPENCV_INC = $(SDK_PATH)/opencv/include
MW_INC = $(MW_PATH)/include
TRACER_INC = $(TRACER_PATH)

LIBS := $(SDK_LIB) $(OPENCV_LIB)
INCLUDES := -I$(SDK_INC)
INCLUDES += -I$(OPENCV_INC)
INCLUDES += -I$(MW_INC)
INCLUDES += -I$(PWD)/modules/include


$(TARGET): $(OBJS)
	$(CXX)  $(LDFLAGS) $(TARGET) $(OBJS)  $(LIB_DIR) $(LIBS)

%.o: %.c
	$(CC) -O3 -fPIC -c $< -o $@ $(INCLUDES) $(CFLAGS)
%.o: %.cpp
	$(CXX) -O3 -fPIC -c $< -o $@ $(INCLUDES) $(CFLAGS)

.PHONY: clean  
clean:
	rm -rf $(OBJS) $(TARGET)
