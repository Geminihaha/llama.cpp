# Qualcomm Adreno OpenCL 백엔드 오류 수정 보고서 (최종본 - 2026-07-02)

이 문서는 Qualcomm Adreno 660 GPU 환경(Termux Android)에서 `llama.cpp` OpenCL 가속을 활성화할 때 발생하던 Flash Attention 커널 컴파일 오류, 런타임 워크그룹 사이즈 오류, 그리고 quantized KV 캐시(`--cache-type-k q8_0`) 활성화 시 발생하는 `clCreateSubBuffer` 크래시 문제를 분석하고 해결한 세부 내역을 기록한 보고서입니다.

---

## 1. 개요 및 발생한 문제

최근 `llama.cpp` OpenCL 백엔드 빌드 및 실행 도중 아래 3가지 핵심 오류로 인해 모델 추론 및 서버 구동이 실패했습니다.

1. **커널 컴파일 오류 (`implicit declaration of function`)**
   * **오류 메시지:**
     ```
     BC-src-code:859:26: error: implicit declaration of function 'sub_group_shuffle' is invalid in C99
     BC-src-code:873:31: error: implicit declaration of function 'sub_group_shuffle_xor' is invalid in C99
     ```
   * **원인:** Qualcomm Adreno GPU 드라이버는 표준 `cl_khr_subgroup_shuffle` 대신 고유 확장인 `cl_qcom_subgroup_shuffle`을 사용하지만, 해당 확장에는 임의 레인 셔플을 제공하는 `qcom_sub_group_shuffle` 함수가 존재하지 않습니다. 이로 인해 커널 내에서 정의되지 않은 함수로 처리되어 컴파일 오류가 발생했습니다.

2. **런타임 워크그룹 사이즈 오류 (`CL_INVALID_WORK_GROUP_SIZE`, `-54`)**
   * **오류 메시지:**
     ```
     ggml_opencl: clEnqueueNDRangeKernel failed with -54. requested local_work_size={256, 1}, kernel max workgroup size=128
     ```
   * **원인:** `DK=256, DV=256` 규격의 Flash Attention 커널은 레지스터 압박 등으로 인해 최대 가능 워크그룹 사이즈가 `128`로 자동 제한되지만, 기존 튜닝 설정은 `local_work_size`로 `256`을 강제 지정하여 실행 도중 크래시가 유발되었습니다.

3. **중복 `clCreateSubBuffer` 크래시 (`CL_INVALID_MEM_OBJECT`, `-38`)**
   * **오류 메시지:**
     ```
     ggml_opencl: clCreateSubBuffer failed for Q8_0 scales with error -38. region.origin=0, region.size=4096, parent=0xb400006e210c51d0, alignment=128
     ```
   * **원인:** 런타임에 Key-Value 캐시 복사 및 추론 시 동일 텐서에 대해 `ggml_backend_opencl_buffer_set_tensor`가 여러 번 실행됩니다. 첫 실행 시 텐서의 `extra` 포인터가 `ggml_tensor_extra_cl`에서 `ggml_tensor_extra_cl_q8_0` 구조체로 변환됩니다. 하지만 구조체가 호환되지 않는 형태로 재정의된 상태에서 두 번째 호출이 일어나면서 `extra_orig->data_device` 멤버가 기형적인 주소(이전 할당된 subbuffer 객체 주소)를 참조하였고, OpenCL 명세상 subbuffer에서 또 subbuffer를 만들 수 없으므로(Nested subbuffer 불가) 드라이버 단에서 `-38` 에러를 반환하며 크래시가 발생했습니다.

---

## 2. 해결 조치 사항

### A. Subgroup Shuffle 로컬 메모리 Fallback 구현
* **해결 조치:**
  * Qualcomm Adreno용 `cl_qcom_subgroup_shuffle`이 활성화된 경우, 하드웨어가 기본 지원하는 [qcom_sub_group_shuffle_xor](file:///data/data/com.termux/files/home/llama.cpp/ggml/src/ggml-opencl/kernels/flash_attn_f32_f16.cl#L7)은 매크로를 통해 맵핑하고, 일반 `sub_group_shuffle`은 공유 로컬 메모리(`l_shuffle_temp`)를 활용하는 [adreno_sub_group_shuffle](file:///data/data/com.termux/files/home/llama.cpp/ggml/src/ggml-opencl/kernels/flash_attn_f32_q4_0.cl#L13) 인라인 함수를 만들어 우회 처리했습니다.
* **대상 파일:**
  * [flash_attn_f32_f16.cl](file:///data/data/com.termux/files/home/llama.cpp/ggml/src/ggml-opencl/kernels/flash_attn_f32_f16.cl)
  * [flash_attn_f32_q4_0.cl](file:///data/data/com.termux/files/home/llama.cpp/ggml/src/ggml-opencl/kernels/flash_attn_f32_q4_0.cl)
  * [flash_attn_f32_q8_0.cl](file:///data/data/com.termux/files/home/llama.cpp/ggml/src/ggml-opencl/kernels/flash_attn_f32_q8_0.cl)

### B. 워크그룹 사이즈 조정 및 튜닝 테이블 패치
* **해결 조치:**
  * [fa_tune.h](file:///data/data/com.termux/files/home/llama.cpp/ggml/src/ggml-opencl/fa_tune.h) 내 Adreno 기본 튜닝 테이블인 [g_fa_dims_adreno_default](file:///data/data/com.termux/files/home/llama.cpp/ggml/src/ggml-opencl/fa_tune.h#L16)에서 `{256, 256}` 규격의 `n_split` 값을 `16`에서 `8`로 축소 조정하여 워크그룹 한계를 초과하지 않도록 패치했습니다.

### C. `clCreateSubBuffer` 중복 생성 방지 가드 구현
* **해결 조치:**
  * [ggml-opencl.cpp](file:///data/data/com.termux/files/home/llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp)의 `ggml_backend_opencl_buffer_set_tensor` 내 `Q8_0` 및 `Q4_0` 처리부에 `ggml_cl_is_q8_0_soa(tensor)` 및 `ggml_cl_is_q4_0_soa(tensor)` 가드 조건을 도입했습니다.
  * 이미 SoA 구조체로 변환된 이력이 있는 텐서의 경우, 기존 할당된 subbuffer(`extra->q`, `extra->d`)를 안전하게 재사용함으로써 잘못된 캐스팅에 의한 메모리 크래시를 원천 차단했습니다.

### D. 런타임 진단 로그 개선
* **해결 조치:**
  * [ggml-opencl.cpp](file:///data/data/com.termux/files/home/llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp)의 [enqueue_ndrange_kernel](file:///data/data/com.termux/files/home/llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp#L883) 및 `clCreateSubBuffer` 실패 분기에 디버깅 시 즉각 오류 원인을 규명할 수 있도록 상세 경고 매크로 로깅을 적용했습니다.

---

## 3. 수정 전후 소스 코드 변경점 (Diff)

### 1) [fa_tune.h](file:///data/data/com.termux/files/home/llama.cpp/ggml/src/ggml-opencl/fa_tune.h)

```diff
-    {256, 256, 16, 16, 16, 0},
+    {256, 256, 16, 16, 8, 0},
```

### 2) [flash_attn_f32_q4_0.cl](file:///data/data/com.termux/files/home/llama.cpp/ggml/src/ggml-opencl/kernels/flash_attn_f32_q4_0.cl) & [flash_attn_f32_q8_0.cl](file:///data/data/com.termux/files/home/llama.cpp/ggml/src/ggml-opencl/kernels/flash_attn_f32_q8_0.cl)

```cl
// 파일 상단 매크로 선언부 수정
#ifdef cl_khr_subgroup_shuffle
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
#elif defined(cl_qcom_subgroup_shuffle)
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
#define sub_group_shuffle(val, id) adreno_sub_group_shuffle((val), (id), l_shuffle_temp)
#define sub_group_shuffle_xor(val, mask) qcom_sub_group_shuffle_xor((val), (mask), CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, (val))
static inline float adreno_sub_group_shuffle(float val, uint id, local float * temp) {
    temp[get_local_id(0)] = val;
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    return temp[(get_local_id(0) - get_sub_group_local_id()) + id];
}
#endif
```

```cl
// 커널 함수 내 로컬 메모리 임시 배열 선언 추가
__kernel void flash_attn_f32_q4_0(
    ...
) {
    const int tid = get_local_id(0);
#if defined(cl_qcom_subgroup_shuffle)
    local float l_shuffle_temp[WG_SIZE];
#endif
    ...
```

### 3) [ggml-opencl.cpp](file:///data/data/com.termux/files/home/llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp) (Q8_0 부분 발췌)

```cpp
    if (tensor->type == GGML_TYPE_Q8_0) {
        // Views share the parent's buffer; parent owns SoA conversion.
        if (tensor->view_src != nullptr || !ggml_is_contiguous(tensor)) {
            return;
        }

        ggml_tensor_extra_cl_q8_0 * extra = nullptr;
        cl_mem data_device = nullptr;
        cl_int err;

        size_t size_d = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        size_t size_q = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*(ggml_blck_size(tensor->type)*sizeof(char));
        GGML_ASSERT(size_d + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");

        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;

        if (ggml_cl_is_q8_0_soa(tensor)) {
            extra = (ggml_tensor_extra_cl_q8_0 *)tensor->extra;
        } else {
            ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
            GGML_ASSERT(extra_orig && "Tensors in OpenCL backend should have been allocated and initialized");

            // Allocate the new extra and create aliases from the original.
            extra = ctx->ggml_opencl_alloc_temp_tensor_extra_q8_0();

            // The original tensor memory is divided into scales and quants, i.e.,
            // we first store scales, then quants.
            cl_buffer_region region;

            // Create subbuffer for scales.
            region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
            region.size = size_d;
            extra->d = clCreateSubBuffer(
                extra_orig->data_device, CL_MEM_READ_WRITE,
                CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
            if (err != CL_SUCCESS) {
                GGML_LOG_ERROR("ggml_opencl: clCreateSubBuffer failed for Q8_0 scales with error %d. region.origin=%zu, region.size=%zu, parent=%p, alignment=%zu\n",
                    err, region.origin, region.size, (void*)extra_orig->data_device, (size_t)backend_ctx->alignment);
            }
            CL_CHECK(err);
            auto previous_origin = region.origin;

            // Create subbuffer for quants.
            region.origin = align_to(previous_origin + size_d, backend_ctx->alignment);
            region.size = size_q;
            extra->q = clCreateSubBuffer(
                extra_orig->data_device, CL_MEM_READ_WRITE,
                CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
            if (err != CL_SUCCESS) {
                GGML_LOG_ERROR("ggml_opencl: clCreateSubBuffer failed for Q8_0 quants with error %d. region.origin=%zu, region.size=%zu, parent=%p, alignment=%zu\n",
                    err, region.origin, region.size, (void*)extra_orig->data_device, (size_t)backend_ctx->alignment);
            }
            CL_CHECK(err);

            tensor->extra = extra;
            ctx->q8_0_soa_tensors.insert(tensor);
        }

        data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);
        CL_CHECK(clEnqueueWriteBuffer(
            queue, data_device, CL_TRUE, 0,
            ggml_nbytes(tensor), data, 0, NULL, NULL));
        ...
```

---

## 4. 최종 결과 확인

* **빌드:** 성공
* **서버 구동 실행 결과:**
  * `Gemma-4` 모델을 로드하고 `top-k` 샘플러 등의 OpenCL 비호환 연산을 CPU로 완만하게 폴백하며 정상 초기화에 성공했습니다.
  * **콘솔 출력 확인:**
    ```
    0.55.545.979 I srv  llama_server: model loaded
    0.55.546.675 I srv  llama_server: listening on http://0.0.0.0:56810
    ```
  * 크래시 없이 포트 `56810`에서 성공적으로 리스닝 대기 모드에 들어감을 확인했습니다.
