# OpenCL Weight Corruption Fix - Changes Summary

Date: 2026-07-02
Branch: fix_opencl_v2x_adreno
Target: ggml/src/ggml-opencl/ggml-opencl.cpp

---

## 1. clReleaseProgram Bulk Release

### Problem
In `load_cl_kernels()`, approximately 50-60 OpenCL program objects were stored as
`backend_ctx->program_*` member variables after kernel creation but were **never released**
with `clReleaseProgram()`. Only a handful of blocks (tri, fill, diag, FA kernels, etc.)
used a local `cl_program prog` variable and released it immediately.

Over multiple kernel compilations, the OpenCL driver heap accumulated these unreleased
program objects, eventually causing `CL_OUT_OF_HOST_MEMORY` (err=-6).

### Fix
Added a bulk release block at the end of `load_cl_kernels()`, right before
`kernels_loaded = true`. All stored program pointers are collected in an array and
released with nullptr-safe iteration:

```cpp
cl_program * programs[] = {
    &backend_ctx->program_add,
    &backend_ctx->program_add_id,
    &backend_ctx->program_argsort_f32_i32,
    &backend_ctx->program_clamp,
    &backend_ctx->program_conv_2d_f16,
    &backend_ctx->program_conv_2d_f16_f32,
    &backend_ctx->program_conv_2d_f32,
    &backend_ctx->program_cvt,
    &backend_ctx->program_diag_mask_inf,
    &backend_ctx->program_div,
    &backend_ctx->program_gelu,
    &backend_ctx->program_gemm_moe_mxfp4_f32,
    &backend_ctx->program_gemv_moe_mxfp4_f32,
    &backend_ctx->program_get_rows,
    &backend_ctx->program_glu,
    &backend_ctx->program_group_norm,
    &backend_ctx->program_im2col_f16,
    &backend_ctx->program_im2col_f32,
    &backend_ctx->program_mul,
    &backend_ctx->program_mul_mat_f16_f32_tiled,
    &backend_ctx->program_mul_mm_f16_f32_kq,
    &backend_ctx->program_mul_mm_f16_f32_kqv,
    &backend_ctx->program_mul_mm_f16_f32_l4_lm,
    &backend_ctx->program_mul_mm_f32_f32_l4_lm,
    &backend_ctx->program_mul_mm_q8_0_f32_l4_lm,
    &backend_ctx->program_mul_mv_f16_f16,
    &backend_ctx->program_mul_mv_f16_f32,
    &backend_ctx->program_mul_mv_f16_f32_1row,
    &backend_ctx->program_mul_mv_f16_f32_l4,
    &backend_ctx->program_mul_mv_f32_f32,
    &backend_ctx->program_mul_mv_id_mxfp4_f32,
    &backend_ctx->program_mul_mv_id_mxfp4_f32_flat,
    &backend_ctx->program_mul_mv_id_q4_0_f32_8x_flat,
    &backend_ctx->program_mul_mv_id_q8_0_f32,
    &backend_ctx->program_mul_mv_id_q8_0_f32_flat,
    &backend_ctx->program_mul_mv_mxfp4_f32,
    &backend_ctx->program_mul_mv_mxfp4_f32_flat,
    &backend_ctx->program_mul_mv_q4_0_f32,
    &backend_ctx->program_mul_mv_q4_0_f32_1d_16x_flat,
    &backend_ctx->program_mul_mv_q4_0_f32_1d_8x_flat,
    &backend_ctx->program_mul_mv_q4_0_f32_8x_flat,
    &backend_ctx->program_mul_mv_q4_0_f32_v,
    &backend_ctx->program_mul_mv_q6_K,
    &backend_ctx->program_mul_mv_q8_0_f32,
    &backend_ctx->program_mul_mv_q8_0_f32_flat,
    &backend_ctx->program_norm,
    &backend_ctx->program_pad,
    &backend_ctx->program_relu,
    &backend_ctx->program_rms_norm,
    &backend_ctx->program_rope,
    &backend_ctx->program_set_rows,
    &backend_ctx->program_sigmoid,
    &backend_ctx->program_silu,
    &backend_ctx->program_softmax_4_f16,
    &backend_ctx->program_softmax_4_f32,
    &backend_ctx->program_softmax_f16,
    &backend_ctx->program_softmax_f32,
    &backend_ctx->program_sub,
    &backend_ctx->program_sum_rows_f32,
    &backend_ctx->program_transpose,
    &backend_ctx->program_tsembd,
    &backend_ctx->program_upscale,
};
for (size_t i = 0; i < sizeof(programs)/sizeof(programs[0]); i++) {
    if (*programs[i]) {
        clReleaseProgram(*programs[i]);
        *programs[i] = nullptr;
    }
}
```

### Safety
- OpenCL spec: "The program object associated with the kernel object can be released
  after the kernel object is created." -- kernel objects retain their compiled code.
- nullptr-safe: programs never assigned (e.g. inside `#ifdef GGML_OPENCL_USE_ADRENO_KERNELS`
  blocks, or conditionally available kernel sources like upscale/tsembd) are skipped.
- Pointers are set to nullptr after release to prevent double-free in future cleanup code.

---

## 2. Adreno Compile Options Optimization

### Problem
On Adreno GPUs (Qualcomm), the OpenCL compiler flags `-cl-unsafe-math-optimizations`
and `-cl-fast-relaxed-math` can cause the shader compiler to consume excessive heap
memory, contributing to `CL_OUT_OF_HOST_MEMORY` (err=-6) during `clBuildProgram()`.

### Fix
In both `load_cl_kernels()` and `load_cl_kernels_argsort()`, added an Adreno-specific
check after the compile_opts string is built:

```cpp
// Adreno drivers can run out of compiler heap with unsafe/fast math flags
if (backend_ctx->gpu_family == GPU_FAMILY::ADRENO) {
    compile_opts.erase(compile_opts.find(" -cl-unsafe-math-optimizations"), strlen(" -cl-unsafe-math-optimizations"));
    compile_opts.erase(compile_opts.find(" -cl-fast-relaxed-math"),         strlen(" -cl-fast-relaxed-math"));
}
```

### What remains
- `-cl-mad-enable` -- kept (safe, low-cost optimization)
- `-cl-finite-math-only` -- kept (safe, enables common algebraic simplifications)
- `-DGGML_OPENCL_USE_ADRENO_KERNELS` -- kept (Adreno-specific kernel dispatch)
- `-qcom-enable-large-buffer` -- kept (Adreno large buffer support)

---

## 3. get_tensor Offset Consistency (Verification Only)

### Scope
Verified that `ggml_backend_opencl_buffer_get_tensor()` handles offsets consistently
with `ggml_backend_opencl_buffer_set_tensor()`.

### Results

| Path | Offset Used | Consistent? |
|------|-------------|-------------|
| Q4_0 AoS (non-SoA) | `extra_aos->offset + tensor->view_offs + offset` | Yes -- same as set_tensor |
| Q4_0 SoA (general) | `offset` (on restore kernel output buffer) | Yes -- restore kernel produces full AoS at offset 0 |
| Q4_0 SoA (Adreno moe) | `offset` | Yes |
| Q4_0 SoA (Adreno noshuffle) | `offset` | Yes |
| Q1_0 general | `offset` | Yes |
| Q1_0 Adreno | `offset` | Yes |
| Q4_1, Q5_0, Q5_1, etc. | `offset` | Yes |

**Conclusion**: No offset mismatch between get_tensor and set_tensor. The subbuffer
fallback path (`cl_create_sub_buffer_fallback`) does not break get_tensor because all
SoA restore kernels read from the correct subbuffer/fallback handles.

---

## 4. Build Verification

- **Build System**: Ninja
- **Result**: 218/218 targets, zero errors
- **OpenCL Library**: `bin/libggml-opencl.so` (5,036,280 bytes)
- **Existing Warnings**: 3 (pre-existing, unrelated)

---

## Files Changed

| File | Lines Changed | Description |
|------|---------------|-------------|
| `ggml/src/ggml-opencl/ggml-opencl.cpp` | +325 / -167 | clReleaseProgram bulk release + Adreno compile_opts + previous subbuffer fallback work |
| `opencl_weight_corruption_fix_progress.md` | Updated | Progress tracking document updated with fix details |

---

## Testing Required (on-device)

1. **Kernel compilation**: Run `llama-server` with `GGML_OPENCL_DEBUG=1` and confirm
   no `err=-6` during `load_cl_kernels`.
2. **Inference accuracy**: Compare first-token logits (CPU vs OpenCL, fixed seed).
3. **Weight integrity**: Run `test-backend-ops` if available.
