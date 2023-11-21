CONFIRM_ENV_VAR = 1
ifeq ($(MAKECMDGOALS), $(filter $(MAKECMDGOALS), install clean))
CONFIRM_ENV_VAR = 0
endif

ifeq ($(CONFIRM_ENV_VAR), 1)
ifndef MW_PATH
$(error "Please set middleware sdk root path to MW_PATH")
endif

ifeq ($(USE_TPU_IVE), ON)
ifndef IVE_PATH
$(error "Please set ive sdk root path to IVE_PATH)
endif
endif

ifndef TPU_PATH
$(error "Please set tpu sdk root path to TPU_PATH)
endif

ifneq ($(CHIP), CV181X)
ifndef CVI_TRACER_PATH
$(error "Please set cvi_tracer sdk root path to CVI_TRACER_PATH)
endif
endif
endif

CHIP ?= cv1835
SDK_VER ?= 64bit

CROSS_COMPILE_32 ?= arm-linux-gnueabihf-
CROSS_COMPILE_64 ?= aarch64-linux-gnu-
CROSS_COMPILE_32_UCLIBC ?= arm-cvitek-linux-uclibcgnueabihf-
CROSS_COMPILE_RISCV64_GLIBC ?= riscv64-unknown-linux-gnu-
CROSS_COMPILE_RISCV64_MUSL ?= riscv64-unknown-linux-musl-

ifeq ($(SDK_VER), 32bit)
ARCH=arm
CROSS_COMPILE = $(CROSS_COMPILE_32)
CFLAGS += -mfloat-abi=hard -mfpu=neon-vfpv4 -march=armv7-a
else ifeq ($(SDK_VER), 64bit)
ARCH=arm64
CROSS_COMPILE = $(CROSS_COMPILE_64)
CFLAGS += -march=armv8-a
else ifeq ($(SDK_VER), uclibc)
ARCH=uclibc
CROSS_COMPILE = $(CROSS_COMPILE_32_UCLIBC)
CFLAGS += -mfloat-abi=hard -mfpu=neon-vfpv4 -march=armv7-a
else ifeq ($(SDK_VER), glibc_riscv64)
ARCH=riscv64
CROSS_COMPILE = $(CROSS_COMPILE_RISCV64_GLIBC)
CFLAGS += -march=rv64imafdcvxthead -mabi=lp64dv
else ifeq ($(SDK_VER), musl_riscv64)
ARCH=riscv64
CROSS_COMPILE = $(CROSS_COMPILE_RISCV64_MUSL)
CFLAGS += -march=rv64imafd
else
$(error "Unknown SDK_VER: $(SDK_VER")")
endif
#####################################################
# Gcc Compiler
#####################################################
CC = $(CROSS_COMPILE)gcc
CXX = $(CROSS_COMPILE)g++
AR = $(CROSS_COMPILE)ar
LD = $(CROSS_COMPILE)ld
STRIP = $(CROSS_COMPILE)strip

#####################################################
# Middleware stuffs
#####################################################
MW_LIB_PATH = $(MW_PATH)/lib
MW_3RD_LIB_PATH = $(MW_LIB_PATH)/3rd
MW_INC_PATH = $(MW_PATH)/include
MW_SAMPLE_INC_PATH = $(MW_PATH)/sample/common
MW_AUDIO_LIBS = -lcvi_audio -lcvi_vqe -lcvi_ssp -ltinyalsa -lcvi_VoiceEngine -lcvi_RES1 -lcli

ifeq ($(CHIP), CV182X)
MW_LIBS = -lsample -lcvi_bin -lcvi_bin_isp -lvpu -lisp -lisp_algo -lvenc -lvdec -lcvi_jpeg -lcvi_vcodec -lae -laf -lawb -lsys -lsns_full -lcvitracer -lini
MW_ISP_INC_PATH = $(MW_INC_PATH)/isp/cv182x
else ifeq ($(CHIP), CV181X)
MW_LIBS = -lsample -lcvi_bin -lcvi_bin_isp -lvpu -lisp -lisp_algo -lvenc -lvdec -lvenc -lae -laf -lawb -lsys -lsns_full -lini -lmisc
MW_ISP_INC_PATH = $(MW_INC_PATH)/isp/cv181x/
else
MW_LIBS = -lsample -lcvi_bin -lcvi_bin_isp -lvpu -lisp -lvenc -lvdec -lcvi_jpeg -lcvi_vcodec -lae -laf -lawb -lsys -lsns_full -lcvitracer -lini
MW_ISP_INC_PATH = $(MW_INC_PATH)/isp/cv183x
endif

#####################################################
# TDL SDK stuffs
#####################################################
SDK_ROOT_PATH ?= $(abspath ../..)
SDK_LIB_PATH = $(SDK_ROOT_PATH)/lib
SDK_INC_PATH = $(SDK_ROOT_PATH)/include
SDK_TDL_INC_PATH = $(SDK_ROOT_PATH)/include/cvi_tdl
SDK_APP_INC_PATH = $(SDK_ROOT_PATH)/include/cvi_tdl_app
SDK_SAMPLE_INC_PATH = $(SDK_ROOT_PATH)/sample/3rd/include
SDK_TDL_LIBS = -lcvi_tdl
SDK_APP_LIBS = -lcvi_tdl_app

#####################################################
# Opencv
#####################################################
OPENCV_INC_PATH = $(TPU_PATH)/opencv/include
OPENCV_LIB_PATH = $(TPU_PATH)/opencv/lib
OPENCV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

#####################################################
# TPU
#####################################################
TPU_LIB_PATH = $(TPU_PATH)/lib
TPU_LIBS = -lcnpy -lcvikernel -lcvimath -lcviruntime -lz -lm

#####################################################
# IVE SDK
#####################################################
ifeq ($(USE_TPU_IVE), ON)
IVE_SDK_LIB_PATH = $(IVE_PATH)/lib
IVE_SDK_INC_PATH = $(IVE_PATH)/include
IVE_LIBS = -lcvi_ive_tpu
CFLAGS += -DUSE_TPU_IVE
else
IVE_SDK_LIB_PATH = $(MW_PATH)/lib
IVE_SDK_INC_PATH = $(MW_PATH)/include
IVE_LIBS = -lcvi_ive
endif

#####################################################
# RTSP stuffs
#####################################################
RTSP_LIB_PATH = $(SDK_ROOT_PATH)/sample/3rd/lib
RTSP_INC_PATH = $(SDK_ROOT_PATH)/sample/3rd/include/cvi_rtsp
RTSP_LIBS = -lcvi_rtsp

#####################################################
# STB
#####################################################
STB_INC_PATH = $(SDK_ROOT_PATH)/sample/3rd/include/stb

#####################################################
# CVI LIB stuffs
#####################################################
MD_INC_PATH = $(SDK_ROOT_PATH)/include/cvi_md
DRAW_RECT_INC_PATH = $(SDK_ROOT_PATH)/include/cvi_draw_rect
PREPROCESS_INC_PATH = $(SDK_ROOT_PATH)/include/cvi_preprocess

#####################################################
# CVI TDL Evaluation
#####################################################
EVA_LIB_PATH = $(SDK_ROOT_PATH)/sample/3rd/lib
EVA_INC_PATH = $(SDK_ROOT_PATH)/sample/3rd/include/cvi_tdl_evaluation
EVA_LIBS = -lcvi_tdl_evaluation

#####################################################
# CVI Tracer
#####################################################
CVI_TRACER_LIB_PATH = $(CVI_TRACER_PATH)/$(ARCH)/lib/3rd
CVI_TRACER_INC_PATH = $(CVI_TRACER_PATH)/include

# OpenCV
# link OpenCV shared object only if platform is cv183X series
ifeq ($(CHIP), $(filter $(CHIP), cv1832 cv1835 cv1838 cv9520 CV183X))
SAMPLE_LIBS_OPENCV += -L$(OPENCV_LIB_PATH) $(OPENCV_LIBS)
endif

CFLAGS += -latomic -fsigned-char -Werror=all -Wno-format-truncation -fdiagnostics-color=always -s 

ifeq ($(DEBUG), 1)
CFLAGS += -g -O0
else
CFLAGS += -O3 -DNDEBUG
endif



ifeq ($(CHIP), CV181X)
CFLAGS += -DCV181X
ifeq ($(SYSTEM_PROCESSOR), RISCV)
CV181X_INC_PATH = ${KERNEL_PATH}/build/${PROJECT_FULLNAME}/riscv/usr/include/
else ifeq ($(SYSTEM_PROCESSOR), ARM)
CV181X_INC_PATH = ${KERNEL_PATH}/build/${PROJECT_FULLNAME}/arm/usr/include/
endif
endif

SAMPLE_LIBS = -L$(MW_LIB_PATH) -L$(MW_3RD_LIB_PATH) $(MW_LIBS) -L$(TPU_LIB_PATH) $(TPU_LIBS) -L$(IVE_SDK_LIB_PATH) $(IVE_LIBS) -L$(SDK_LIB_PATH) -lpthread

ifneq ($(CHIP), CV181X)
SAMPLE_LIBS += -L$(CVI_TRACER_LIB_PATH)
CFLAGS += -I$(CVI_TRACER_INC_PATH)
else
CFLAGS += -I$(CV181X_INC_PATH)
endif

SAMPLE_LIBS_OPENCV += $(OPENCV_LIBS) -L$(OPENCV_LIB_PATH)

CFLAGS += -I$(SDK_INC_PATH) \
          -I$(SDK_TDL_INC_PATH) \
	  -I$(SDK_SAMPLE_INC_PATH) \
	  -I$(MW_INC_PATH) \
	  -I$(MD_INC_PATH) \
	  -I$(DRAW_RECT_INC_PATH) \
	  -I$(PREPROCESS_INC_PATH) \
	  -I$(MW_SAMPLE_INC_PATH) \
	  -I$(MW_ISP_INC_PATH) \
	  -I$(IVE_SDK_INC_PATH) \
	  -I$(OPENCV_INC_PATH) \
	  -I$(EVA_INC_PATH)


ifneq (,$(findstring middleware/v2,$(MW_PATH)))
    CFLAGS += -D_MIDDLEWARE_V2_
endif

ifeq ($(CONFIRM_ENV_VAR), 1)
$(info ---------------------------------------)
$(info CHIP: $(CHIP))
$(info SDK_VER: $(SDK_VER))
$(info TDL SDK library path: $(SDK_LIB_PATH))
$(info TDL SDK include path: $(SDK_INC_PATH))
$(info Middleware include path: $(MW_INC_PATH))
$(info Middleware library path: $(MW_LIB_PATH))
$(info IVE library path: $(IVE_SDK_LIB_PATH))
$(info IVE include path: $(IVE_SDK_INC_PATH))
$(info TPU library path: $(TPU_LIB_PATH))
$(info CFLAGS: $(CFLAGS))
$(info CC: $(CC))
$(info CXX: $(CXX))
$(info USE_TPU_IVE: $(USE_TPU_IVE))
$(info ---------------------------------------)
endif

TARGETS := $(shell find . -type f -name 'sample_lib_*.cpp' -exec basename {} .cpp ';')

.PHONY : all clean install

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

install:
	mkdir -p ../tmp_install/bin
	cp $(TARGETS) ../tmp_install/bin

sample_lib_cvimd: sample_lib_cvimd.cpp
	$(CXX) $(DEPFLAGS) $(CFLAGS) $(SAMPLE_LIBS) -lcvi_md -o $@ $^

sample_lib_draw_rect: sample_lib_draw_rect.cpp
	$(CXX) $(DEPFLAGS) $(CFLAGS) $(SAMPLE_LIBS) $(SAMPLE_LIBS_OPENCV) -lcvi_draw_rect -o $@ $^

sample_lib_preprocess: sample_lib_preprocess.cpp
	$(CXX) $(DEPFLAGS) $(CFLAGS) $(SAMPLE_LIBS) -lcvi_preprocess -o $@ $^

