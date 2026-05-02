# tensor3d_conv3d_q

Quantized Conv2D applied across all output channels (Conv3D).

## Function Signature

```c
MyTensor *tensor3d_conv3d_q(
    const MyTensor  *input,
    const MyTensor  *filters,       /* array [n_filters], each [fH, fW, inC] */
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
```

Returns a newly allocated `MyTensor [out_h, out_w, n_filters]`. Caller must `tensor_free()` it.

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `input` | `MyTensor*` | Input tensor, shape `[H, W, inC]` |
| `filters` | `MyTensor[]` | Array of `n_filters` filters, each shape `[fH, fW, inC]` |
| `n_filters` | `int32_t` | Number of output channels |
| `bias` | `int8_t[]` | Quantized bias, one value per output channel |
| `padding` | `int32_t` | Symmetric zero-pad pixels on each side |
| `dilation_h/w` | `int32_t` | Dilation factor for height / width |
| `stride_h/w` | `int32_t` | Stride for height / width |
| `relu6` | `int` | Non-zero: apply ReLU6 activation |
| `scale_input` | `double` | `S_x` — input quantization scale |
| `scale_output` | `double` | `S_y` — output quantization scale |
| `scale_filter` | `double[]` | `S_w^(k)` — per-channel filter scale |
| `scale_bias` | `double` | `S_b` — bias quantization scale |
| `zp_input` | `int8_t` | `Z_x` — input zero point |
| `zp_output` | `int8_t` | `Z_y` — output zero point |
| `zp_bias` | `int8_t` | `Z_b` — bias zero point |

## Algorithm

### Conv3D = Conv2D × n_filters

```
for k in 0..n_filters:
    output[:, :, k] = conv2d(input, filter[k], bias[k])
```

### Conv2D (per output channel k)

For each output pixel `(oh, ow)`:

```
acc = Σ_{r,s,c}  (input[oh*stride_h + r*dil_h, ow*stride_w + s*dil_w, c] - Z_x)
                 * filter[k][r, s, c]
```

Out-of-bounds input positions (padding region) contribute 0.

### Quantization Formula

$$q_y^{(k)} = \text{clamp}\!\left(\;\text{round}\!\left(M^{(k)} \cdot \left(\text{acc} + b_q^{(k)}\right)\right) + Z_y,\; \text{act\_min},\; \text{act\_max}\right)$$

| Symbol | Formula | Notes |
|---|---|---|
| $M^{(k)}$ | $S_x \cdot S_w^{(k)} / S_y$ | Per-channel scale multiplier |
| $b_q^{(k)}$ | $\text{round}\!\left(\dfrac{S_b}{S_x \cdot S_w^{(k)}} \cdot (q_b^{(k)} - Z_b)\right)$ | Pre-scaled bias (int32) |
| $Z_w$ | $= 0$ | Weight zero point fixed (symmetric quantization) |

### Activation Clamp

| Mode | `act_min` | `act_max` |
|---|---|---|
| None | `-128` | `127` |
| ReLU6 | `max(-128, Z_y)` | `min(127, Z_y + round(6 / S_y))` |

### Output Size

```
out_h = (H + 2*padding - dilation_h*(fH-1) - 1) / stride_h + 1
out_w = (W + 2*padding - dilation_w*(fW-1) - 1) / stride_w + 1
```

## Fixed-Point Implementation (16×16 hardware)

### quantize_multiplier

Decomposes `M^(k)` (double) into a 15-bit integer `mult` and right-shift `shift`:

```
M  ≈  mult * 2^(-shift)
```

Computed once per output channel in software (double precision). `mult` fits in `int16_t`.

### apply_mult_q

Applies the 15-bit multiplier to the 32-bit accumulator using **two 16×16 multiplications**:

```
acc = acc_hi * 2^16 + acc_lo          (split into upper/lower 16 bits)

mult * acc = (mult * acc_hi) << 16
           + (mult * acc_lo)
           + correction               (if acc_lo sign bit represents unsigned magnitude)
```

Both `mult × acc_hi` and `mult × acc_lo` are 16×16 → int32. Result accumulated in int64, then right-shifted by `shift` with round-to-nearest.

### Per-pixel loop (no float)

```
acc  = Σ (x[c] - zp_in) * w[c]       integer accumulation, int32
acc += bias_q[k]                      pre-scaled bias, int32
r    = apply_mult_q(mult, shift, acc) two 16x16 multiplications
r   += zp_output
r    = clamp(r, act_min, act_max)
```
