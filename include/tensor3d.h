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

/* Quantized depthwise-separable style Conv2D over all output channels.
 *
 * Implements (per output channel k, per output pixel oh/ow):
 *   acc   = Σ_{r,s,c} (input[oh*sh+r*dh, ow*sw+s*dw, c] - zp_in) * filter[k][r,s,c]
 *           + round(scale_bias / (scale_input*scale_filter[k]) * (bias[k] - zp_bias))
 *   out[oh,ow,k] = clamp(round(M*acc) + zp_out, act_min, act_max)
 *   where M = scale_input * scale_filter[k] / scale_output
 *
 * All per-pixel arithmetic uses only 16x16 integer multiplications.
 * Bias pre-scaling and M decomposition use software double arithmetic (once).
 *
 * Padding: symmetric, user-specified (pixels on each side).
 * Weight zero point is fixed at 0 per the quantization scheme.
 * relu6: act_max = clamp(zp_out + round(6/scale_output), -128, 127)
 *        act_min = max(-128, zp_out)
 *
 * Returns a newly allocated tensor [out_h, out_w, n_filters]; caller must
 * tensor_free() it.  Returns NULL on allocation failure.
 *
 * out_h = (input->h + 2*padding - dilation_h*(filter_h-1) - 1) / stride_h + 1
 */
MyTensor *tensor3d_conv3d_q(
    const MyTensor  *input,
    const MyTensor  *filters,       /* array [n_filters], each MyTensor [fH,fW,inC] */
    int32_t          n_filters,
    const int8_t    *bias,          /* [n_filters] */
    int32_t          padding,
    int32_t          dilation_h,
    int32_t          dilation_w,
    int32_t          stride_h,
    int32_t          stride_w,
    int              relu6,
    double           scale_input,
    double           scale_output,
    const double    *scale_filter,  /* [n_filters] per output channel */
    double           scale_bias,
    int8_t           zp_input,
    int8_t           zp_output,
    int8_t           zp_bias);
#endif
