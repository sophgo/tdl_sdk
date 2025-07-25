#include <stdint.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "ccl.hpp"
#include "utils/tdl_log.hpp"
#define CC_MAX_NUM_LABELS 200
#define MAX_CC_OBJECTS 200

#define CC_SUPER_PIXEL_H 2
#define CC_SUPER_PIXEL_W 2
#define CC_SCAN_WINDOW_H 2
#define CC_SCAN_WINDOW_W 2
#define CC_FG_SUPER_PIX_THD 3
#define BLOCK_SIZE 2

typedef struct CCTag {
  int maskWidth;
  int maskHeight;
  int ccSuperPixelSize;
  int superPixMapW;
  int superPixMapH;

  int numTotalObj;

  int maxID;

  unsigned char *dataSuperPixMap;
  unsigned char *dataSuperPixFG;
  unsigned char *eqCCLabelArray;

  int numObjects;
  int *boundingBoxesTmp;
  int *boundingBoxes;

} CCLType;

void *createConnectInstance() {
  CCLType *p_inst = new CCLType();
  memset(p_inst, 0, sizeof(CCLType));
  return p_inst;
}

void initConnectedComponent(CCLType *ccGst, int width, int height) {
  int superPixMapW, superPixMapH;

  ccGst->maskWidth = width;
  ccGst->maskHeight = height;

  ccGst->superPixMapW = width / CC_SUPER_PIXEL_W;
  ccGst->superPixMapH = height / CC_SUPER_PIXEL_H;

  superPixMapW = ccGst->superPixMapW;
  superPixMapH = ccGst->superPixMapH;

  ccGst->dataSuperPixMap = new unsigned char[superPixMapW * superPixMapH];

  memset(ccGst->dataSuperPixMap, 0, superPixMapH * superPixMapW);

  ccGst->dataSuperPixFG = new unsigned char[superPixMapH * superPixMapW];
  memset(ccGst->dataSuperPixFG, 0, superPixMapH * superPixMapW);

  ccGst->eqCCLabelArray = new unsigned char[CC_MAX_NUM_LABELS + 1];

  ccGst->boundingBoxesTmp = new int[MAX_CC_OBJECTS * 5];
  ccGst->boundingBoxes = new int[MAX_CC_OBJECTS * 5];
}

void releaseConnectedComponent(CCLType *ccGst) {
  if (ccGst->maskWidth > 0 && ccGst->maskHeight > 0) {
    delete[] ccGst->dataSuperPixMap;
    delete[] ccGst->dataSuperPixFG;
    delete[] ccGst->eqCCLabelArray;
    delete[] ccGst->boundingBoxesTmp;
    delete[] ccGst->boundingBoxes;
  }

  memset(ccGst, 0, sizeof(CCLType));
}

bool clusterBox(int *p_boxes, int ci, int i) {
  // check if intersect
  int inter_R0 = std::max(p_boxes[5 * i + 1], p_boxes[5 * ci + 1]);
  int inter_C0 = std::max(p_boxes[5 * i + 2], p_boxes[5 * ci + 2]);

  int inter_R1 = std::min(p_boxes[5 * i + 3], p_boxes[5 * ci + 3]);
  int inter_C1 = std::min(p_boxes[5 * i + 4], p_boxes[5 * ci + 4]);

  int interw = inter_C1 - inter_C0;
  int interh = inter_R1 - inter_R0;

  if (interw <= 0 || interh <= 0) return false;

  int out_R0 = std::min(p_boxes[5 * i + 1], p_boxes[5 * ci + 1]);
  int out_C0 = std::min(p_boxes[5 * i + 2], p_boxes[5 * ci + 2]);

  int out_R1 = std::max(p_boxes[5 * i + 3], p_boxes[5 * ci + 3]);
  int out_C1 = std::max(p_boxes[5 * i + 4], p_boxes[5 * ci + 4]);
  p_boxes[5 * ci + 1] = out_R0;
  p_boxes[5 * ci + 2] = out_C0;
  p_boxes[5 * ci + 3] = out_R1;
  p_boxes[5 * ci + 4] = out_C1;
  return true;
}
int filterInsideBoxes(int *p_boxes, int num_src) {
  // sort with box area
  struct area_info {
    int area;
    int index;
  };
  std::vector<area_info> box_areas;
  for (int i = 0; i < num_src; i++) {
    int R0 = p_boxes[5 * i + 1];
    int C0 = p_boxes[5 * i + 2];
    int R1 = p_boxes[5 * i + 3];
    int C1 = p_boxes[5 * i + 4];
    int area = (R1 - R0) * (C1 - C0);
    area_info a_info;
    a_info.area = area;
    a_info.index = i;
    box_areas.push_back(a_info);
  }

  std::sort(
      box_areas.begin(), box_areas.end(),
      [](const area_info &a, const area_info &b) { return a.area > b.area; });

  for (size_t i = 1; i < box_areas.size(); i++) {
    int indi = box_areas[i].index;
    // std::cout<<"indi:"<<indi<<",area:"<<box_areas[i].area<<std::endl;
    for (size_t c = 0; c < i; c++) {
      int indc = box_areas[c].index;
      if (p_boxes[5 * indc] < 0) continue;
      if (clusterBox(p_boxes, indc, indi)) {
        p_boxes[5 * indi] = -1;
        break;
      }
    }
  }
  int ctFinal = 0;
  for (int i = 0; i < num_src; i++) {
    if (p_boxes[5 * i] > 0) {
      memcpy(p_boxes + 5 * ctFinal, p_boxes + 5 * i, 5 * sizeof(int));
      ctFinal++;
    }
  }
  return ctFinal;
}
int dumpIveImageFrame(const std::string &filepath, uint8_t *ptr_img, int w,
                      int h, int wstride) {
  FILE *fp = fopen(filepath.c_str(), "wb");
  if (fp == nullptr) {
    return -1;
  }

  fwrite(&w, sizeof(uint32_t), 1, fp);
  fwrite(&h, sizeof(uint32_t), 1, fp);

  std::cout << "width:" << w << ",height:" << h << ",stride:" << wstride
            << std::endl;
  fwrite(ptr_img, w * h, 1, fp);

  // for(int j = 0 ; j < h; j++){
  //     uint8_t *ptrj = ptr + j*wstride;
  //     std::cout<<"write r:"<<j<<std::endl;
  //     fwrite(ptrj,w,1,fp);
  // }
  std::cout << "toclose:" << filepath << std::endl;
  fclose(fp);

  std::cout << "closed:" << filepath << std::endl;
  return 0;
}
int *extractConnectedComponent(unsigned char *p_fg_mask, int width, int height,
                               int wstride, int area_thresh, void *p_cc_inst,
                               int *p_num_boxes) {
  int r, c, i, j, rBlk, cBlk;
  int xA, xB, xC, xD;
  int cumulatedBlkSum, superPixMapW, superPixMapH;
  int ctFGPixels, imgDataWStep;
  int lbVal, currLabel, R0, C0, R1, C1, objectSize;
  int tmpMaxLabel;

  unsigned char *ptrImgData;
  unsigned char *ptrImgData0;
  unsigned char *ptrSuperPixMap;
  unsigned char *ptrSuperPixMap0;
  unsigned char *ptrSuperPixFG;
  unsigned char *ptrSuperPixFG0;
  unsigned char *ptrLabelMap;
  unsigned char *ptrLabelMap0;
  unsigned char *eqLabelArray;
  unsigned char *ptrUCHAR;

  int *boundingBoxes;
  int *boundingBoxesFinal;
  int ctFinal;

  CCLType *ccGst = (CCLType *)p_cc_inst;
  if (ccGst->maskWidth != width || ccGst->maskHeight != height) {
    LOGI("allocate ccl,w:%d,h:%d\n", width, height);

    releaseConnectedComponent(ccGst);
    initConnectedComponent(ccGst, width, height);
    // std::cout<<"to allocate
    // ccl,w:"<<width<<",height:"<<height<<",maskw:"<<ccGst->maskWidth<<std::endl;
  }

  // imgW = ccGst->maskWidth;
  superPixMapW = ccGst->superPixMapW;
  superPixMapH = ccGst->superPixMapH;

  ptrSuperPixMap0 = ccGst->dataSuperPixMap;
  ptrSuperPixFG0 = ccGst->dataSuperPixFG;
  eqLabelArray = ccGst->eqCCLabelArray;

  ptrImgData0 = p_fg_mask;

  tmpMaxLabel = 0;

  /* reset foreground super pixels.*/
  memset(ccGst->dataSuperPixMap, 0,
         superPixMapH * superPixMapW * sizeof(unsigned char));
  memset(ccGst->dataSuperPixFG, 0,
         superPixMapH * superPixMapW * sizeof(unsigned char));

  ptrUCHAR = ccGst->eqCCLabelArray;

  for (i = 0; i < (CC_MAX_NUM_LABELS + 1); i += 1) {
    *ptrUCHAR = i;
    ptrUCHAR += 1;
  }

  ccGst->numTotalObj = 0;

  /***********************************************/
  /*	Sum up foreground count within suer pixels.*/
  imgDataWStep = (CC_SUPER_PIXEL_H * wstride);

  for (rBlk = 0; rBlk < superPixMapH; rBlk += 1) {
    for (cBlk = 0; cBlk < superPixMapW; cBlk += 1) {
      ptrImgData =
          ptrImgData0 + (rBlk * imgDataWStep) + (cBlk * CC_SUPER_PIXEL_W);

      cumulatedBlkSum = 0;

      for (r = 0; r < CC_SUPER_PIXEL_H; r += 1) {
        for (c = 0; c < CC_SUPER_PIXEL_W; c += 1) {
          if ((*ptrImgData) > 0) cumulatedBlkSum += 1;

          ptrImgData += 1;
        } /* c */

        ptrImgData += (wstride - CC_SUPER_PIXEL_W);
      } /* r */

      *(ptrSuperPixFG0 + (rBlk * superPixMapW) + cBlk) = cumulatedBlkSum;
    } /*end of: for ( cBlk )*/
  }   /*end of: for ( rBlk )*/

  /**************************/
  /* Scan image for objects */
  /**************************/

  for (r = 0; r < (superPixMapH - CC_SCAN_WINDOW_H); r += 1) {
    for (c = 0; c < (superPixMapW - CC_SCAN_WINDOW_W); c += 1) {
      /* sum up pixel in scanning window.*/

      ctFGPixels = 0;

      ptrSuperPixFG = ptrSuperPixFG0 + (r * superPixMapW) + c;

      for (i = 0; i < CC_SCAN_WINDOW_H; i += 1) {
        for (j = 0; j < CC_SCAN_WINDOW_W; j += 1) {
          ctFGPixels += (*ptrSuperPixFG);
          ptrSuperPixFG += 1;

        } /* col: j */

        ptrSuperPixFG += (superPixMapW - CC_SCAN_WINDOW_W);
      } /* row: i */

      /* mark foreground.*/

      if (ctFGPixels > CC_FG_SUPER_PIX_THD) {
        ptrSuperPixMap = ptrSuperPixMap0 + (r * superPixMapW) + c;

        for (i = 0; i < CC_SCAN_WINDOW_H; i += 1) {
          for (j = 0; j < CC_SCAN_WINDOW_W; j += 1) {
            *(ptrSuperPixMap) = 255;
            ptrSuperPixMap += 1;
          }

          ptrSuperPixMap += (superPixMapW - CC_SCAN_WINDOW_W);
        }

      } /*end of: mark foreground.*/

    } /*end of: for ( c )*/
  }   /*end of: for ( r )*/

  // now the map is saved in ptrSuperPixMap
  /*	Connected component labeling.*/

  currLabel = 1;
  // dump_ive_image_frame("/mnt/data/admin1_data/alios_test/md/fg.bin",ptrSuperPixMap0,ccGst->superPixMapW,ccGst->superPixMapH,ccGst->superPixMapW);
  /* put label in [*ptrSuperPixFG].*/
  memset(ccGst->dataSuperPixFG, 0,
         superPixMapH * superPixMapW * sizeof(unsigned char));
  ptrLabelMap0 = ccGst->dataSuperPixFG;

  /* the 1st pass for setting label and flooding label.*/
  for (r = 1; r < superPixMapH; r += 1) {
    ptrSuperPixMap = ptrSuperPixMap0 + (r * superPixMapW) + 1;
    ptrLabelMap = ptrLabelMap0 + (r * superPixMapW) + 1;

    for (c = 1; c < superPixMapW; c += 1) {
      if ((*ptrSuperPixMap) > 0) {
        /*	xB	xC	xD	*/
        /*	xA	x		*/
        xA = *(ptrLabelMap - 1);
        xB = *(ptrLabelMap - 1 - superPixMapW);
        xC = *(ptrLabelMap - superPixMapW);
        xD = *(ptrLabelMap - superPixMapW + 1);

        lbVal = CC_MAX_NUM_LABELS;

        if (xA > 0) lbVal = xA;
        if ((xB > 0) && (xB < lbVal)) lbVal = xB;
        if ((xC > 0) && (xC < lbVal)) lbVal = xC;
        if ((xD > 0) && (xD < lbVal)) lbVal = xD;

        if (lbVal < CC_MAX_NUM_LABELS) {
          /* make connection and flood label.*/
          if (xA > lbVal) {
            *(eqLabelArray + xA) = *(eqLabelArray + lbVal);
          }
          if (xB > lbVal) {
            *(eqLabelArray + xB) = *(eqLabelArray + lbVal);
          }
          if (xC > lbVal) {
            *(eqLabelArray + xC) = *(eqLabelArray + lbVal);
          }
          if (xD > lbVal) {
            *(eqLabelArray + xD) = *(eqLabelArray + lbVal);
          }
        } else {
          /* new label.*/
          lbVal = currLabel;
          if (currLabel <
              (CC_MAX_NUM_LABELS -
               1)) {  // the maximum label would be CC_MAX_NUM_LABELS-1
            currLabel += 1;
          }
        }

        /* save label for this pixel.*/
        *(ptrLabelMap) = lbVal;

      } /*end of: if foreground.*/

      ptrSuperPixMap += 1;
      ptrLabelMap += 1;
    } /* col */
  }   /* row */

  /* the 2nd pass to flush with equivalent labels.*/
  for (r = 1; r < superPixMapH; r += 1) {
    ptrLabelMap = ptrLabelMap0 + (r * superPixMapW) + 1;

    for (c = 1; c < superPixMapW; c += 1) {
      lbVal = (*ptrLabelMap);

      if (lbVal > 0) {
        /* equivalent label.*/
        lbVal = *(eqLabelArray + lbVal);
        *ptrLabelMap = lbVal;

        // find the maximal label.
        if (lbVal > tmpMaxLabel) {
          tmpMaxLabel = lbVal;
        }

        // display only:
        *(ccGst->dataSuperPixMap + (r * superPixMapW) + c) = lbVal;

      } /*end of: if foreground.*/

      ptrLabelMap += 1;
    } /* col */
  }   /* row */

  // this would not happen
  if (tmpMaxLabel > CC_MAX_NUM_LABELS) {
    printf("ccl error maxlable:%d\n", tmpMaxLabel);
    *p_num_boxes = 0;
    return ccGst->boundingBoxes;
  }
  ccGst->maxID = tmpMaxLabel;

  if (tmpMaxLabel > 0) {
    tmpMaxLabel = tmpMaxLabel;
  }

  // determine the bounding boxes of each object
  boundingBoxes = ccGst->boundingBoxesTmp;
  for (i = 0; i <= ccGst->maxID; i++) {
    boundingBoxes[5 * i + 0] = 0;
    boundingBoxes[5 * i + 1] = 10000;
    boundingBoxes[5 * i + 2] = 10000;
    boundingBoxes[5 * i + 3] = -10000;
    boundingBoxes[5 * i + 4] = -10000;
  }

  for (r = 1; r < superPixMapH; r += 1) {
    ptrLabelMap = ccGst->dataSuperPixMap + (r * superPixMapW) + 1;

    for (c = 1; c < superPixMapW; c += 1) {
      lbVal = *ptrLabelMap;
      ptrLabelMap++;

      if (lbVal >= 1) {
        if (lbVal > (CC_MAX_NUM_LABELS - 1)) lbVal = CC_MAX_NUM_LABELS - 1;

        lbVal = lbVal - 1;
        boundingBoxes[5 * lbVal] = boundingBoxes[5 * lbVal] + 1;

        if (r < boundingBoxes[5 * lbVal + 1]) boundingBoxes[5 * lbVal + 1] = r;
        if (c < boundingBoxes[5 * lbVal + 2]) boundingBoxes[5 * lbVal + 2] = c;
        if (r > boundingBoxes[5 * lbVal + 3]) boundingBoxes[5 * lbVal + 3] = r;
        if (c > boundingBoxes[5 * lbVal + 4]) boundingBoxes[5 * lbVal + 4] = c;
      }
    }
  }

  // clean up: remove small ones and remove overlap
  boundingBoxes = ccGst->boundingBoxesTmp;
  boundingBoxesFinal = ccGst->boundingBoxes;
  ctFinal = 0;

  // remove overlapped boxes
  int area_super_thresh = area_thresh / BLOCK_SIZE / BLOCK_SIZE;
  for (i = 0; i < tmpMaxLabel; i++) {
    objectSize = boundingBoxes[5 * i + 0];
    if (objectSize > area_super_thresh) {
      R0 = boundingBoxes[5 * i + 1] * BLOCK_SIZE;
      C0 = boundingBoxes[5 * i + 2] * BLOCK_SIZE;
      R1 = boundingBoxes[5 * i + 3] * BLOCK_SIZE;
      C1 = boundingBoxes[5 * i + 4] * BLOCK_SIZE;
      int area = (C1 - C0) * (R1 - R0);
      if (area < area_thresh) continue;
      boundingBoxesFinal[5 * ctFinal + 0] = objectSize;
      boundingBoxesFinal[5 * ctFinal + 1] = R0;
      boundingBoxesFinal[5 * ctFinal + 2] = C0;
      boundingBoxesFinal[5 * ctFinal + 3] = R1;
      boundingBoxesFinal[5 * ctFinal + 4] = C1;
      ctFinal++;
    }
  }

  ctFinal = filterInsideBoxes(boundingBoxesFinal, ctFinal);
  ctFinal = filterInsideBoxes(boundingBoxesFinal, ctFinal);
  ccGst->maxID = ctFinal;
  ccGst->numObjects = ctFinal;
  *p_num_boxes = ccGst->numObjects;
  return ccGst->boundingBoxes;
} /*end of: void extractConnectedComponent() | connected component labeling.*/

void destroyConnectedComponent(void *ccGst) {
  if (ccGst == NULL) return;
  CCLType *p_inst = (CCLType *)ccGst;
  releaseConnectedComponent(p_inst);
  delete p_inst;
}
