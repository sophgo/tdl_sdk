SHELL = /bin/bash
# include ./Makefile.param

ifndef MW_PATH
$(error "Please set middleware sdk root path to MW_PATH")
endif

ifndef IVE_PATH
$(error "Please set rtsp sdk root path to IVE_PATH)
endif

ifndef TPU_PATH
$(error "Please set rtsp sdk root path to TPU_PATH)
endif

CHIP ?= cv1835
SDK_VER ?= 64bit

# Middleware stuffs
#####################################################
MW_LIB_PATH = $(MW_PATH)/lib
MW_3RD_LIB_PATH = $(MW_PATH)/lib/3rd
MW_INC_PATH = $(MW_PATH)/include
MW_SAMPLE_INC_PATH = $(MW_PATH)/sample/common
MW_AUDIO_LIBS = -lcvi_audio -lcvi_vqe -ltinyalsa -lcvi_VoiceEngine -lcvi_RES -lnanomsg

ifeq ($(CHIP), $(filter $(CHIP), cv1832 cv1835 cv1838 cv9520))
MW_LIBS = -lsample -lvpu -lisp -lvenc -lvdec -lcvi_jpeg -lcvi_vcodec -lae -laf -lawb -lsys -lsns_full -lcvitracer -lini
MW_ISP_INC_PATH = $(MW_INC_PATH)/isp/cv183x
else
MW_LIBS = -lsample -lvpu -lisp -lisp_algo -lvenc -lvdec -lcvi_jpeg -lcvi_vcodec -lae -laf -lawb -lsys -lsns_full -lcvitracer -lini
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

# OpenCV
#####################################################
OPENCV_LIB_PATH = $(TPU_PATH)/opencv/lib

# IVE
#####################################################
IVE_SDK_LIB_PATH = $(IVE_PATH)/lib
IVE_SDK_INC_PATH = $(IVE_PATH)/include


CFLAGS += -Os -std=gnu11 -march=armv7-a -ffunction-sections -fdata-sections -Wno-pointer-to-int-cast -fsigned-char -mfloat-abi=hard -mfpu=neon-vfpv4 -Os -std=gnu11 -march=armv7-a -ffunction-sections -fdata-sections -Wno-pointer-to-int-cast -fsigned-char -mfloat-abi=hard -mfpu=neon-vfpv4 -O3 -DNDEBUG -fsigned-char -Werror=all -fdiagnostics-color=always -std=c11 -s -MD
ifeq ($(DEBUG), 1)
CFLAGS += -g -O0
STRIPSYMBOL =
else
CFLAGS += -O3
STRIPSYMBOL = @$(STRIP) $@
endif

CROSS_COMPILE_32 ?= arm-linux-gnueabihf-
CROSS_COMPILE_64 ?= aarch64-linux-gnu-

ifeq ($(SDK_VER), 32bit)
ARCH=arm
CROSS_COMPILE = $(CROSS_COMPILE_32)
else
ARCH=arm64
CROSS_COMPILE = $(CROSS_COMPILE_64)
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
AISDK_LIBS = -lcviai_core -lcviai_service
TPU_LIBS = -lcnpy  -lcvikernel  -lcvimath  -lcviruntime  -lz
IVE_LIBS = -lcvi_ive_tpu
OPENCV_LIBS = -lopencv_core -lopencv_imgproc

SRCS := $(wildcard $(PWD)/*.c)
OBJS := $(SRCS:%.c=%.o)
DEPS = $(SRCS:.c=.d)

SAMPLE_LIBS = $(AISDK_LIBS) $(TPU_LIBS) $(IVE_LIBS) $(OPENCV_LIBS) -L$(AISDK_LIB_PATH) -L$(MW_LIB_PATH) -L$(TPU_LIB_PATH) -L$(IVE_SDK_LIB_PATH) -L$(OPENCV_LIB_PATH) -L$(MW_3RD_LIB_PATH) $(MW_LIBS) -lpthread
SAMPLE_VI_LIBS = $(SAMPLE_LIBS) $(RTSP_LIBS) -L$(RTSP_LIB_PATH)

CFLAGS += -I$(AISDK_INC_PATH) -I$(MW_INC_PATH)  -I$(RTSP_INC_PATH) -I$(MW_SAMPLE_INC_PATH) -I$(MW_ISP_INC_PATH) -I$(RTSP_INC_PATH) -I$(IVE_SDK_INC_PATH)

TARGETS = sample_init \
			sample_read_dt \
			sample_read_esc \
			sample_read_fq_tracking \
			sample_read_fr \
			sample_read_fr_custom \
			sample_read_lpdr \
			sample_read_mot_1 \
			sample_read_mot_2 \
			sample_read_mot_3 \
			sample_read_pose \
			sample_read_td \
			sample_vi_dms \
			sample_vi_fall \
			sample_vi_fd \
			sample_vi_fq \
			sample_vi_fq_tracking \
			sample_vi_lpdr \
			sample_vi_mask_fr \
			sample_vi_md \
			sample_vi_od \
			sample_vi_people_tracking \
			sample_vi_tracking


.PHONY : clean all

all: $(TARGETS)

clean:
	rm -f $(OBJS) $(TARGETS) $(DEPS)

$(PWD)/%.o: $(PWD)/%.c
	$(CC) $(DEPFLAGS) $(CFLAGS) -o $@  -c $<

sample_read_esc: $(PWD)/sample_read_esc.o
	$(CC) $(SAMPLE_LIBS) $(MW_AUDIO_LIBS) -o $@ $^

sample_vi_%: $(PWD)/sample_vi_%.o $(PWD)/vi_vo_utils.o
	$(CC) $(SAMPLE_VI_LIBS)  -o $@ $^

sample_read_%: $(PWD)/sample_read_%.o
	$(CC) $(SAMPLE_LIBS)  -o $@ $^

sample_init: $(PWD)/sample_init.o
	$(CC) $(SAMPLE_LIBS)  -o $@ $^
