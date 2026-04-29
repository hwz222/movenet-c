#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
