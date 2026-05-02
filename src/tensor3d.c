#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor3d.h"

MyTensor *tensor_create(int32_t h, int32_t w, int32_t c)
{
    MyTensor *t = (MyTensor *)malloc(sizeof(MyTensor));
    if (!t) return NULL;

    t->h = h;
    t->w = w;
    t->c = c;

    t->data = (int8_t *)calloc((size_t)h * w * c, sizeof(int8_t));
    if (!t->data) {
        free(t);
        return NULL;
    }
    return t;
}

void tensor_free(MyTensor *t)
{
    if (!t) return;
    free(t->data);
    free(t);
}

/* linear index for HWC layout */
static inline int32_t _idx(const MyTensor *t, int32_t h, int32_t w, int32_t c)
{
    return h * (t->w * t->c) + w * t->c + c;
}

int8_t tensor_get(const MyTensor *t, int32_t h, int32_t w, int32_t c)
{
    return t->data[_idx(t, h, w, c)];
}

void tensor_set(MyTensor *t, int32_t h, int32_t w, int32_t c, int8_t val)
{
    t->data[_idx(t, h, w, c)] = val;
}

int8_t *tensor_at(MyTensor *t, int32_t h, int32_t w, int32_t c)
{
    return &t->data[_idx(t, h, w, c)];
}

size_t tensor_size(const MyTensor *t)
{
    return (size_t)t->h * t->w * t->c;
}

void tensor_fill(MyTensor *t, int8_t val)
{
    memset(t->data, val, tensor_size(t));
}

void tensor_quantize_u8(MyTensor *t, const uint8_t *src)
{
    size_t n = tensor_size(t);
    for (size_t i = 0; i < n; i++)
        t->data[i] = (int8_t)((int16_t)src[i] - 128);
}

void tensor_add_scalar(MyTensor *t, int8_t scalar)
{
    size_t n = tensor_size(t);
    for (size_t i = 0; i < n; i++) {
        int16_t v = (int16_t)t->data[i] + scalar;
        if (v > 127)       v = 127;
        else if (v < -128) v = -128;
        t->data[i] = (int8_t)v;
    }
}

void tensor_print(const MyTensor *t)
{
    printf("MyTensor [H=%d W=%d C=%d]\n", t->h, t->w, t->c);
    for (int32_t h = 0; h < t->h; h++) {
        printf("  h=%d:\n", h);
        for (int32_t w = 0; w < t->w; w++) {
            printf("    w=%d: [", w);
            for (int32_t c = 0; c < t->c; c++) {
                printf("%4d", tensor_get(t, h, w, c));
                if (c < t->c - 1) printf(",");
            }
            printf("]\n");
        }
    }
}

/* ================================================================
 *  Conv2D helpers
 * ================================================================ */

/* Apply a 15-bit fixed-point multiplier to a 32-bit accumulator using
 * two 16x16 hardware multiplications.
 * Computes round( mult * acc * 2^(-shift) ).
 *
 * acc is split into hi (upper 16 bits) and lo (lower 16 bits).
 * Each part is multiplied by mult (16x16 → int32), then recombined.
 * A sign correction is applied because acc_lo is signed int16 but
 * represents unsigned 16-bit data (upper bit is magnitude, not sign). */
static int32_t apply_mult_q(int16_t mult, int shift, int32_t acc)
{
    int16_t acc_hi = (int16_t)(acc >> 16);
    int16_t acc_lo = (int16_t)(acc & 0xFFFF);

    int32_t prod_hi = (int32_t)mult * (int32_t)acc_hi;  /* 16x16 → 32 */
    int32_t prod_lo = (int32_t)mult * (int32_t)acc_lo;  /* 16x16 → 32 */

    int64_t r = ((int64_t)prod_hi << 16) + (int64_t)prod_lo;
    if (acc_lo < 0) r += (int64_t)mult << 16;  /* correct unsigned lower bits */

    if (shift > 0) { r += 1LL << (shift - 1); r >>= shift; }
    else if (shift < 0) { r <<= (-shift); }
    return (int32_t)r;
}

/* Accumulate the dot product for one output pixel (oh, ow) and one filter.
 * Σ_{r,s,c} (input[ih, iw, c] - zp_in) * filter[r, s, c]
 * Out-of-bounds input pixels (padding region) contribute 0. */
static int32_t conv2d_accum(
    const MyTensor *input,
    const MyTensor *filter,
    int32_t oh,       int32_t ow,
    int32_t stride_h, int32_t stride_w,
    int32_t dil_h,    int32_t dil_w,
    int32_t pad_h,    int32_t pad_w,
    int8_t  zp_in)
{
    int32_t acc = 0;
    for (int32_t fh = 0; fh < filter->h; fh++) {
        int32_t ih = oh * stride_h + fh * dil_h - pad_h;
        if (ih < 0 || ih >= input->h) continue;
        for (int32_t fw = 0; fw < filter->w; fw++) {
            int32_t iw = ow * stride_w + fw * dil_w - pad_w;
            if (iw < 0 || iw >= input->w) continue;
            const int8_t *x = &input->data[(ih * input->w + iw) * input->c];
            const int8_t *w = &filter->data[(fh * filter->w + fw) * filter->c];
            for (int32_t c = 0; c < filter->c; c++)
                acc += ((int32_t)x[c] - (int32_t)zp_in) * (int32_t)w[c];
        }
    }
    return acc;
}

/* ================================================================ */

void quantize_multiplier(float M, int16_t *mult, int *shift)
{
    if (M == 0.0f) { *mult = 0; *shift = 0; return; }

    /* Use double precision (software, runs once) for accurate int16 result.
       frexp: M = mantissa * 2^exp,  mantissa in [0.5, 1.0) */
    int exp;
    double mantissa = frexp((double)M, &exp);

    /* 15 bits of precision: mult * 2^(-shift) ≈ M  →  shift = 15 - exp */
    *shift = 15 - exp;
    int32_t q = (int32_t)llround(mantissa * (double)(1 << 15));

    /* rounding can reach exactly 2^15 — normalise one step */
    if (q == (1 << 15)) { q >>= 1; (*shift)--; }

    *mult = (int16_t)q;
}

void tensor_add_q(MyTensor       *out,
                  const MyTensor *a,
                  const MyTensor *b,
                  float scale_out,   int8_t zp_out,
                  float scale_a,     int8_t zp_a,
                  float scale_b,     int8_t zp_b)
{
    int16_t mult1, mult2;
    int     shift1, shift2;
    quantize_multiplier(scale_a / scale_out, &mult1, &shift1);
    quantize_multiplier(scale_b / scale_out, &mult2, &shift2);

    /* Normalise to a common shift (software, once before the loop).
       Right-shift the multiplier that is "ahead" — it stays int16 so the
       per-element multiply remains a hardware 16x16 operation.
       The other multiplier is used as-is (no precision loss). */
    int common_shift = (shift1 > shift2) ? shift1 : shift2;
    int16_t m1 = (int16_t)(mult1 >> (common_shift - shift1));
    int16_t m2 = (int16_t)(mult2 >> (common_shift - shift2));

    size_t n = tensor_size(out);
    for (size_t i = 0; i < n; i++) {
        int16_t da = (int16_t)a->data[i] - (int16_t)zp_a;
        int16_t db = (int16_t)b->data[i] - (int16_t)zp_b;

        /* Two 16x16 -> 32 multiplications, accumulated before single rounding. */
        int32_t acc = (int32_t)m1 * (int32_t)da
                    + (int32_t)m2 * (int32_t)db;

        int32_t r;
        if (common_shift > 0) {
            acc += 1 << (common_shift - 1);     /* round-to-nearest */
            r = acc >> common_shift;
        } else {
            r = acc;
        }
        r += (int32_t)zp_out;

        if      (r >  127) r =  127;
        else if (r < -128) r = -128;
        out->data[i] = (int8_t)r;
    }
}

MyTensor *tensor3d_conv3d_q(
    const MyTensor  *input,
    const MyTensor  *filters,
    int32_t          n_filters,
    const int8_t    *bias,
    int32_t          padding,
    int32_t          dilation_h,
    int32_t          dilation_w,
    int32_t          stride_h,
    int32_t          stride_w,
    int              relu6,
    double           scale_input,
    double           scale_output,
    const double    *scale_filter,
    double           scale_bias,
    int8_t           zp_input,
    int8_t           zp_output,
    int8_t           zp_bias)
{
    int32_t fh     = filters[0].h;
    int32_t fw     = filters[0].w;
    int32_t out_h  = (input->h + 2*padding - dilation_h*(fh-1) - 1) / stride_h + 1;
    int32_t out_w  = (input->w + 2*padding - dilation_w*(fw-1) - 1) / stride_w + 1;

    MyTensor *out = tensor_create(out_h, out_w, n_filters);
    if (!out) return NULL;

    /* Activation clamp bounds (int8 space) */
    int32_t act_min = -128;
    int32_t act_max =  127;
    if (relu6) {
        act_min = (int32_t)zp_output;
        act_max = (int32_t)zp_output + (int32_t)llround(6.0 / scale_output);
        if (act_min < -128) act_min = -128;
        if (act_max >  127) act_max =  127;
    }

    /* Per-channel setup (software, runs once per channel) ──────────────
     * M^(k) = scale_input * scale_filter[k] / scale_output
     * bias_q^(k) = round( scale_bias / (scale_input * scale_filter[k])
     *                     * (bias[k] - zp_bias) )
     * ----------------------------------------------------------------- */
    int16_t *mult   = (int16_t *)malloc((size_t)n_filters * sizeof(int16_t));
    int     *shift  = (int     *)malloc((size_t)n_filters * sizeof(int));
    int32_t *bias_q = (int32_t *)malloc((size_t)n_filters * sizeof(int32_t));
    if (!mult || !shift || !bias_q) {
        free(mult); free(shift); free(bias_q);
        tensor_free(out); return NULL;
    }

    for (int32_t k = 0; k < n_filters; k++) {
        double M = scale_input * scale_filter[k] / scale_output;
        quantize_multiplier((float)M, &mult[k], &shift[k]);
        double bias_ratio = scale_bias / (scale_input * scale_filter[k]);
        bias_q[k] = (int32_t)llround(bias_ratio * (double)(bias[k] - zp_bias));
    }

    /* Main loops ──────────────────────────────────────────────────────── */
    for (int32_t oh = 0; oh < out_h; oh++) {
        for (int32_t ow = 0; ow < out_w; ow++) {
            for (int32_t k = 0; k < n_filters; k++) {

                /* Integer accumulation — no multiply hardware needed */
                int32_t acc = conv2d_accum(
                    input, &filters[k],
                    oh, ow,
                    stride_h, stride_w,
                    dilation_h, dilation_w,
                    padding, padding,
                    zp_input);

                /* Add pre-scaled bias */
                acc += bias_q[k];

                /* Scale to output domain: two 16x16 multiplications */
                int32_t r = apply_mult_q(mult[k], shift[k], acc)
                          + (int32_t)zp_output;

                if      (r > act_max) r = act_max;
                else if (r < act_min) r = act_min;

                out->data[(oh * out_w + ow) * n_filters + k] = (int8_t)r;
            }
        }
    }

    free(mult); free(shift); free(bias_q);
    return out;
}