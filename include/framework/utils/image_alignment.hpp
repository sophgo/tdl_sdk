#ifndef IMAGE_ALIGNMENT_HPP
#define IMAGE_ALIGNMENT_HPP
#include <cstdint>

// transform[6] = [cosθ, sinθ, tx,  -sinθ, cosθ,ty]
int tdl_get_similarity_transform_matrix(const float* src_pts_xy,
                                        const float* dst_pts_xy,
                                        int num_points,
                                        float* transform);

void tdl_warp_affine(const unsigned char* src_data,
                     const float* affine_transform,
                     unsigned int src_step,
                     int src_width,
                     int src_height,
                     unsigned char* dst_data,
                     unsigned int dst_step,
                     int dst_width,
                     int dst_height);

int32_t tdl_face_warp_affine(const unsigned char* src_data,
                             unsigned int src_step,
                             int src_width,
                             int src_height,
                             unsigned char* dst_data,
                             unsigned int dst_step,
                             int dst_width,
                             int dst_height,
                             const float* src_pts5_xy);

int32_t tdl_license_plate_warp_affine(const unsigned char* src_data,
                                      unsigned int src_step,
                                      int src_width,
                                      int src_height,
                                      unsigned char* dst_data,
                                      unsigned int dst_step,
                                      int dst_width,
                                      int dst_height,
                                      const float* src_pts4_xy);
#endif
