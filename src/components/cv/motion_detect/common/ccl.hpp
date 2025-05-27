#ifndef FILE_CCL_HPP
#define FILE_CCL_HPP

void* createConnectInstance();

int* extractConnectedComponent(unsigned char* p_fg_mask, int width, int height,
                               int wstride, int area_thresh, void* p_cc_inst,
                               int* p_num_boxes);
void destroyConnectedComponent(void* p_cc_inst);

#endif
