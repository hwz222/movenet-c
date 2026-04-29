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
void tensor_add_scalar(MyTensor *t, int8_t scalar);
#endif
