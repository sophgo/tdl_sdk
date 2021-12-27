SHELL = /bin/bash
# include ./Makefile.param

ifndef MW_PATH
$(error "Please set middleware sdk root path to MW_PATH")
endif

ifndef IVE_PATH
$(error "Please set ive sdk root path to IVE_PATH)
endif

ifndef TPU_PATH
$(error "Please set tpu sdk root path to TPU_PATH)
endif

ifndef CVI_TRACER_PATH
$(error "Please set cvi_tracer sdk root path to CVI_TRACER_PATH)
endif

CHIP ?= cv1835
SDK_VER ?= 64bit

CROSS_COMPILE_32 ?= arm-linux-gnueabihf-
CROSS_COMPILE_64 ?= aarch64-linux-gnu-
CROSS_COMPILE_32_UCLIBC ?= arm-cvitek-linux-uclibcgnueabihf-
ifeq ($(SDK_VER), 32bit)
ARCH=arm
CROSS_COMPILE = $(CROSS_COMPILE_32)
LINK_OPENCV = 0
CFLAGS += -mfloat-abi=hard -mfpu=neon-vfpv4 -march=armv7-a
else ifeq ($(SDK_VER), 64bit)
ARCH=arm64
CROSS_COMPILE = $(CROSS_COMPILE_64)
LINK_OPENCV = 1
CFLAGS += -march=armv8-a
else ifeq ($(SDK_VER), uclibc)
ARCH=uclibc
CROSS_COMPILE = $(CROSS_COMPILE_32_UCLIBC)
LINK_OPENCV = 0
CFLAGS += -mfloat-abi=hard -mfpu=neon-vfpv4 -march=armv7-a
else
$(error "Unknown SDK_VER: $(SDK_VER")")
endif

# Middleware stuffs
#####################################################
MW_LIB_PATH = $(MW_PATH)/lib
MW_3RD_LIB_PATH = $(MW_PATH)/lib/3rd
MW_INC_PATH = $(MW_PATH)/include
MW_SAMPLE_INC_PATH = $(MW_PATH)/sample/common
MW_AUDIO_LIBS = -lcvi_audio -lcvi_vqe -ltinyalsa -lcvi_VoiceEngine -lcvi_RES1

ifeq ($(CHIP), $(filter $(CHIP), cv1832 cv1835 cv1838 cv9520))
MW_LIBS = -lsample -lcvi_bin -lcvi_bin_isp -lvpu -lisp -lvenc -lvdec -lcvi_jpeg -lcvi_vcodec -lae -laf -lawb -lsys -lsns_full -lcvitracer -lini
MW_ISP_INC_PATH = $(MW_INC_PATH)/isp/cv183x
else
MW_LIBS = -lsample -lcvi_bin -lcvi_bin_isp -lvpu -lisp -lisp_algo -lvenc -lvdec -lcvi_jpeg -lcvi_vcodec -lae -laf -lawb -lsys -lsns_full -lcvitracer -lini
MW_ISP_INC_PATH = $(MW_INC_PATH)/isp/cv182x
endif
#####################################################


# AISDK stuffs
#####################################################
AISDK_ROOT_PATH ?= ..
AISDK_LIB_PATH = $(AISDK_ROOT_PATH)/lib
AISDK_INC_PATH = $(AISDK_ROOT_PATH)/include/cviai
#####################################################


# RTSP stuffs
#####################################################
RTSP_LIB_PATH = $(AISDK_ROOT_PATH)/lib
RTSP_INC_PATH = $(AISDK_ROOT_PATH)/include/cvi_rtsp
#####################################################

# TPU
#####################################################
TPU_LIB_PATH = $(TPU_PATH)/lib

# IVE
#####################################################
IVE_SDK_LIB_PATH = $(IVE_PATH)/lib
IVE_SDK_INC_PATH = $(IVE_PATH)/include

# cvi tracer
#####################################################
CVI_TRACER_LIB_PATH = $(CVI_TRACER_PATH)/$(ARCH)/lib/3rd
CVI_TRACER_INC_PATH = $(CVI_TRACER_PATH)/include

# OpenCV
ifeq ($(LINK_OPENCV), 1)
OPENCV_LIB_PATH = $(TPU_PATH)/opencv/lib
OPENCV_LIBS = -lopencv_core -lopencv_imgproc
endif

# 3rd party
#####################################################
THIRD_PARTY_PATH = $(PWD)/3rd_party
STB_URL = http://10.58.65.3:8480/kevin.yang/stb.git
STB_PATH = $(THIRD_PARTY_PATH)/stb
THIRD_PARTY_INC_FLAGS = -I$(STB_PATH)

CFLAGS += -std=gnu11 -Wno-pointer-to-int-cast -fsigned-char -fsigned-char -Werror=all -fdiagnostics-color=always -s 

ifeq ($(DEBUG), 1)
CFLAGS += -g -O0
STRIPSYMBOL =
else
CFLAGS += -O3 -DNDEBUG
STRIPSYMBOL = @$(STRIP) $@
endif


## GCC COMPILER ##
CC = $(CROSS_COMPILE)gcc
CXX = $(CROSS_COMPILE)g++
AR = $(CROSS_COMPILE)ar
LD = $(CROSS_COMPILE)ld
STRIP = $(CROSS_COMPILE)strip


$(info ---------------------------------------)
$(info CHIP: $(CHIP))
$(info SDK_VER: $(SDK_VER))
$(info Middleware include path: $(MW_INC_PATH))
$(info Middleware library path: $(MW_LIB_PATH))
$(info AI SDK library path: $(AISDK_LIB_PATH))
$(info AI SDK include path: $(AISDK_INC_PATH))
$(info IVE library path: $(IVE_SDK_LIB_PATH))
$(info IVE include path: $(IVE_SDK_INC_PATH))
$(info TPU library path: $(TPU_LIB_PATH))
$(info CFLAGS: $(CFLAGS))
$(info CC: $(CC))
$(info CXX: $(CXX))
$(info ---------------------------------------)


RTSP_LIBS = -lcvi_rtsp
AISDK_BASIC_LIBS = -lcviai_core -lcviai_service
AISDK_APP_LIBS = -lcviai_app

TPU_LIBS = -lcnpy  -lcvikernel  -lcvimath  -lcviruntime  -lz
IVE_LIBS = -lcvi_ive_tpu

SRCS := $(wildcard $(PWD)/*.c)
OBJS := $(SRCS:%.c=%.o)
DEPS = $(SRCS:.c=.d)

SAMPLE_LIBS = $(AISDK_BASIC_LIBS) $(TPU_LIBS) $(IVE_LIBS) -L$(AISDK_LIB_PATH) -L$(MW_LIB_PATH) -L$(TPU_LIB_PATH) -L$(IVE_SDK_LIB_PATH) -L$(CVI_TRACER_LIB_PATH) -L$(MW_3RD_LIB_PATH) $(MW_LIBS) -lpthread

ifeq ($(LINK_OPENCV), 1)
SAMPLE_LIBS += $(OPENCV_LIBS) -L$(OPENCV_LIB_PATH)
endif

SAMPLE_VI_LIBS = $(SAMPLE_LIBS) $(RTSP_LIBS) -L$(RTSP_LIB_PATH)
SAMPLE_APP_LIBS = $(SAMPLE_LIBS) $(RTSP_LIBS) -L$(RTSP_LIB_PATH) -lcviai_app

CFLAGS += -I$(AISDK_INC_PATH) -I$(MW_INC_PATH)  -I$(RTSP_INC_PATH) -I$(MW_SAMPLE_INC_PATH) -I$(MW_ISP_INC_PATH) -I$(RTSP_INC_PATH) -I$(IVE_SDK_INC_PATH) -I$(CVI_TRACER_INC_PATH)
CFLAGS += $(THIRD_PARTY_INC_FLAGS)

TARGETS_VI_SAMPLE := $(shell find . -type f -name 'sample_vi_*.c' -exec basename {} .c ';')
TARGETS_AUDIO_SAMPLE := $(shell find . -type f -name 'sample_aud_*.c' -exec basename {} .c ';')
TARGETS_IMAGE_SAMPLE := $(shell find . -type f -name 'sample_read_*.c' -exec basename {} .c ';')
TARGETS_APP_SAMPLE := $(shell find . -type f -name 'sample_app_*.c' -exec basename {} .c ';')

TARGETS = sample_init \
		$(TARGETS_VI_SAMPLE) \
		$(TARGETS_AUDIO_SAMPLE) \
		$(TARGETS_IMAGE_SAMPLE) \
		$(TARGETS_APP_SAMPLE)

.PHONY : clean all

all: $(TARGETS)

clean:
	rm -f $(OBJS) $(TARGETS) $(DEPS)
	rm -rf $(THIRD_PARTY_PATH)

$(PWD)/%.o: $(PWD)/%.c | $(THIRD_PARTY_PATH)
	$(CC) $(DEPFLAGS) $(CFLAGS) -o $@  -c $<

$(THIRD_PARTY_PATH):
	git clone	$(STB_URL) $(STB_PATH)

sample_aud_%: $(PWD)/sample_aud_%.o $(PWD)/sample_utils.o
	$(CC) $(SAMPLE_LIBS) $(MW_AUDIO_LIBS)  -o $@ $^

sample_app_%: $(PWD)/sample_app_%.o $(PWD)/vi_vo_utils.o
	$(CC) $(SAMPLE_APP_LIBS)  -o $@ $^

sample_vi_%: $(PWD)/sample_vi_%.o $(PWD)/vi_vo_utils.o $(PWD)/sample_utils.o
	$(CC) $(SAMPLE_VI_LIBS)  -o $@ $^

sample_read_%: $(PWD)/sample_read_%.o $(PWD)/sample_utils.o
	$(CC) $(SAMPLE_LIBS)  -o $@ $^

sample_init: $(PWD)/sample_init.o
	$(CC) $(SAMPLE_LIBS)  -o $@ $^
