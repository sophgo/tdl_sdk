PWD := $(shell pwd)

CC = $(TOOLCHAIN_PATH)/aarch64-linux-gnu-gcc
CXX := $(TOOLCHAIN_PATH)/aarch64-linux-gnu-g++

TARGET := sample_fd

SRC_C += $(shell find ./ -name '*.c')
OBJS := $(SRC_C:%.c=%.o)
OBJS += $(SRC_CPP:%.cpp=%.o)

MLIR_INC ?= $(SDK_PATH)/include
MLIR_LIB ?= -L$(SDK_PATH)/lib -lcviruntime -lcvikernel -lcnpy -lcvimath -lz
OPENCV_LIB = -L$(SDK_PATH)/opencv/lib -lopencv_imgcodecs -lopencv_core -lopencv_imgproc
OPENCV_INC = $(SDK_PATH)/opencv/include

SNS_LIB = -lsns_full
LIBS += -L$(MW_PATH)/lib -lsample -lvpu -lisp -lvenc -lcvi_vcodec -lbmjpu -lawb -lm -lsys -lpthread $(SNS_LIB) $(MLIR_LIB) $(OPENCV_LIB)
LIBS += -lcvimath
LIBS += -L$(PWD)/../ -lcvialgo
#LIBS += -L$(TRACER_PATH)/build -lcvitracer


INCLUDES = -I$(PWD)/../modules/include
INCLUDES += -I$(MW_PATH)/include
INCLUDES += -I$(TRACER_PATH)
INCLUDES += -I$(OPENCV_INC)
INCLUDES += -I$(MLIR_INC)
INCLUDES += -I$(MW_PATH)/sample/common

CLFAGS += -fPIC -pthread -fpermissive

.PHONY: default clean install

all: default

default: $(OBJS)
	$(CC) -fPIC -o $(TARGET) $(OBJS) $(LIB_DIR) $(LIBS)

%.o: %.c $(DEPS)
	$(CC) -c $< -o $@ $(INCLUDES) $(CFLAGS)
%.o: %.cpp $(DEPS)
	$(CXX) -c $< -o $@ $(INCLUDES) $(CFLAGS)

clean:
	rm -rf $(OBJS) $(TARGET)

install: 
	$(shell if [ ! -d ../install ]; then mkdir ../install;mkdir ../install/lib;fi)
	cp ./$(TARGET) ../install


