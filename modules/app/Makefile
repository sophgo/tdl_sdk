SHELL = /bin/bash
# include ./Makefile.param

CHECK_SDK = 1
ifeq ($(MAKECMDGOALS), $(filter $(MAKECMDGOALS), install clean))
CHECK_SDK = 0
endif
ifndef MAKECMDGOALS
CHECK_SDK = 1
endif

ifeq ($(CHECK_SDK), 1)
ifndef MW_PATH
$(error "Please set middleware sdk root path to MW_PATH")
endif

ifeq ($(USE_TPU_IVE), ON)
ifndef IVE_PATH
$(error "Please set rtsp sdk root path to IVE_PATH)
endif
endif

ifndef TPU_PATH
$(error "Please set rtsp sdk root path to TPU_PATH)
endif
endif

CHIP ?= cv1835
SDK_VER ?= 64bit

CROSS_COMPILE_32 ?= arm-linux-gnueabihf-
CROSS_COMPILE_64 ?= aarch64-linux-gnu-
CROSS_COMPILE_32_UCLIBC ?= arm-cvitek-linux-uclibcgnueabihf-
CROSS_COMPILE_RISCV64_GLIBC ?= riscv64-unknown-linux-gnu-
CROSS_COMPILE_RISCV64_MUSL ?= riscv64-unknown-linux-musl-

# link OpenCV shared object only if platform is cv183X series
ifeq ($(CHIP), $(filter $(CHIP), cv1832 cv1835 cv1838 cv9520 CV183X))
LINK_OPENCV = 1
else
LINK_OPENCV = 0
endif

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

# Middleware stuffs
#####################################################
MW_INC_PATH = $(MW_PATH)/include
#####################################################


# AISDK stuffs
#####################################################
AISDK_ROOT_PATH ?= ../..
AISDK_LIB_PATH = $(AISDK_ROOT_PATH)/lib
AISDK_INC_PATH = $(AISDK_ROOT_PATH)/include/cviai
#####################################################


# IVE
#####################################################
IVE_SDK_INC_PATH = $(IVE_PATH)/include

ifeq ($(CHIP), MARS)
CFLAGS += -DMARS
ifeq ($(SYSTEM_PROCESSOR), RISCV)
MARS_INC_PATH = ${KERNEL_PATH}/build/${PROJECT_FULLNAME}/riscv/usr/include/
else ifeq ($(SYSTEM_PROCESSOR), ARM)
MARS_INC_PATH = ${KERNEL_PATH}/build/${PROJECT_FULLNAME}/arm/usr/include/
endif
endif

CFLAGS += -std=gnu11 -Wno-pointer-to-int-cast -fsigned-char -Werror=all -fdiagnostics-color=always -s 

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

AISDK_LIBS = -lcviai

AI_APP_LIBS = $(AISDK_LIBS) -L$(AISDK_LIB_PATH) -lpthread

CFLAGS += -I$(AISDK_INC_PATH) -I$(MW_INC_PATH)

ifeq ($(USE_TPU_IVE), ON)
CFLAGS += -I$(IVE_SDK_INC_PATH)
endif

ifeq ($(CHIP), MARS)
CFLAGS += -I$(MARS_INC_PATH)
endif

CFLAGS += -Wl,-gc-sections

ifneq (,$(findstring middleware/v2,$(MW_PATH)))
    CFLAGS += -D_MIDDLEWARE_V2_
endif

ifeq ($(CHECK_SDK), 1)
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
$(info USE_TPU_IVE: $(USE_TPU_IVE))
$(info ---------------------------------------)
endif


.PHONY : all clean install

all: libcviai_app.so

face_capture.o: face_capture/face_capture.c
	$(CC) $(DEPFLAGS) $(CFLAGS) -fPIC -o $@ -c $^

person_capture.o: person_capture/person_capture.c
	$(CC) $(DEPFLAGS) $(CFLAGS) -fPIC -o $@ -c $^

cviai_app.o: cviai_app.c
	$(CC) $(DEPFLAGS) $(CFLAGS) -fPIC -o $@ -c $^

libcviai_app.so: cviai_app.o face_capture.o person_capture.o
	$(CC) $(CFLAGS) $(AI_APP_LIBS) $^ -o $@ -shared

clean:
	rm -f libcviai_app.so cviai_app.o face_capture.o

install:
	mkdir -p ../../tmp_install/lib
	cp libcviai_app.so ../../tmp_install/lib
