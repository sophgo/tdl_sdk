#ifndef __CVI_COMM_H__
#define __CVI_COMM_H__

#if (defined(__CV181X__) || defined(__CV180X__)) && !defined(_MIDDLEWARE_V3_)
#include <linux/cvi_comm_gdc.h>
#include <linux/cvi_comm_video.h>
#include <linux/cvi_comm_vpss.h>
#include <linux/cvi_errno.h>
#include <linux/cvi_math.h>
#else
#include <cvi_comm_gdc.h>
#include <cvi_comm_video.h>
#include <cvi_comm_vpss.h>
#include <cvi_errno.h>
#include <cvi_math.h>
#endif
#include <cvi_comm_vb.h>
#include <cvi_sys.h>
#include <cvi_type.h>
#endif