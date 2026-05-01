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