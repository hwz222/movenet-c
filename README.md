# movenet-c

C implementation of tensor operations for MoveNet quantized inference.

## Project Structure

```
movenet-c/
├── include/
│   └── tensor3d.h        # Public API
├── src/
│   └── tensor3d.c        # Tensor operations
└── tests/
    ├── hex_compare.c                      # File comparison tool
    ├── run_all_tb.sh                      # Master test runner
    ├── tb_tensor3d_tensor_quantize_u8.c   # TB: quantize_u8
    └── tb_tensor3d_tensor_add_q.c         # TB: add_q
```

## Tensor

HWC memory layout (`data[h*W*C + w*C + c]`), `int8_t` storage.

```c
typedef struct {
    int8_t  *data;
    int32_t  h, w, c;
} MyTensor;
```

## API

### Create / Free
```c
MyTensor *tensor_create(int32_t h, int32_t w, int32_t c);
void      tensor_free(MyTensor *t);
```

### Access
```c
int8_t  tensor_get(const MyTensor *t, int32_t h, int32_t w, int32_t c);
void    tensor_set(MyTensor *t, int32_t h, int32_t w, int32_t c, int8_t val);
int8_t *tensor_at(MyTensor *t, int32_t h, int32_t w, int32_t c);
size_t  tensor_size(const MyTensor *t);
```

### Operations

#### tensor_quantize_u8
Convert a `uint8_t` buffer to `int8_t` by subtracting 128.
```c
void tensor_quantize_u8(MyTensor *t, const uint8_t *src);
```

#### tensor_add_scalar
Element-wise add a scalar with saturation clamp to `[-128, 127]`.
```c
void tensor_add_scalar(MyTensor *t, int8_t scalar);
```

#### tensor_add_q
Element-wise quantized add of two same-shape tensors.
```c
void tensor_add_q(MyTensor       *out,
                  const MyTensor *a,
                  const MyTensor *b,
                  float scale_out, int8_t zp_out,
                  float scale_a,   int8_t zp_a,
                  float scale_b,   int8_t zp_b);
```
Formula:
```
out[i] = clamp( M1*(a[i]-zp_a) + M2*(b[i]-zp_b) + zp_out, -128, 127 )
  where M1 = scale_a / scale_out
        M2 = scale_b / scale_out
```
All per-element arithmetic uses `int16 x int16 -> int32` multiplications.
Both products are accumulated before a single rounding right-shift to avoid double-rounding error.

#### quantize_multiplier
Decompose a float scale ratio into a 15-bit fixed-point multiplier and shift count.
Computed in software (double precision, runs once). Result fits in `int16` for hardware 16×16 multiply.
```c
void quantize_multiplier(float M, int16_t *mult, int *shift);
// M * x  ≈  (mult * x) >> shift
```

## Tests

Test benches read/write space-separated hex files and compare against a reference.

**Run all tests:**
```bash
bash tests/run_all_tb.sh
```

**Adding a new TB:**
1. Write `tests/tb_tensor3d_<function>.c` — reads inputs, calls function, writes output hex
2. Add a `compile_tb` line in the `COMPILE TBs` section of `run_all_tb.sh`
3. Add a run + `compare` block in the `RUN TEST CASES` section

## Build

```bash
make
```
Output binary: `build/circle_tool`
