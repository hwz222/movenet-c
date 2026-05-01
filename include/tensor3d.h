#ifndef TENSOR3D_H
#define TENSOR3D_H

#include <stddef.h>
#include <stdint.h>

/* HWC memory layout: data[h*W*C + w*C + c] */
typedef struct {
    int8_t  *data;
    int32_t  h; /* height */
    int32_t  w; /* width */
    int32_t  c; /* channels */
} MyTensor;

MyTensor *tensor_create(int32_t h, int32_t w, int32_t c);
void      tensor_free(MyTensor *t);

int8_t  tensor_get(const MyTensor *t, int32_t h, int32_t w, int32_t c);
void    tensor_set(MyTensor *t, int32_t h, int32_t w, int32_t c, int8_t val);
int8_t *tensor_at(MyTensor *t, int32_t h, int32_t w, int32_t c);

size_t tensor_size(const MyTensor *t);
void   tensor_fill(MyTensor *t, int8_t val);
void   tensor_print(const MyTensor *t);

void tensor_quantize(MyTensor *t, float scale, int8_t zero_point);
void tensor_quantize_u8(MyTensor *t, const uint8_t *src);
void tensor_add_scalar(MyTensor *t, int8_t scalar);

/* Decompose a float multiplier M into a 15-bit fixed-point integer and a
   right-shift count so that  M * x  ≈  (mult * x) >> shift  (rounded).
   Computed in software (double precision) so mult fits in int16 for
   hardware 16x16 multiply. */
void quantize_multiplier(float M, int16_t *mult, int *shift);

/* Element-wise quantized add:
   out[i] = clamp( M1*(a[i]-zp_a) + M2*(b[i]-zp_b) + zp_out, -128, 127 )
   where M1 = scale_a/scale_out,  M2 = scale_b/scale_out.
   All arithmetic is done in fixed-point (no float per element). */
void tensor_add_q(MyTensor       *out,
                  const MyTensor *a,
                  const MyTensor *b,
                  float scale_out,   int8_t zp_out,
                  float scale_a,     int8_t zp_a,
                  float scale_b,     int8_t zp_b);
#endif
