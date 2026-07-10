# Qualcomm Adreno OpenCL 가중치 틀어짐(Weight Corruption) 수정 진행 보고서
작성일: 2026-07-02

---

## 1. 문제 요약

### 발생 환경
- **기기**: Qualcomm Adreno 660 GPU (Android, Termux)
- **프로젝트**: llama.cpp OpenCL 백엔드
- **빌드 경로**: `/data/data/com.termux/files/home/llama.cpp/build_opencl/build-android/`
- **관련 파일**: `ggml/src/ggml-opencl/ggml-opencl.cpp`

### 핵심 증상
1. 모델 로드 이후 추론 결과가 완전히 비정상 (garbage output)
2. 특정 레이어의 가중치 값이 엉뚱한 메모리 위치에서 읽힘
3. 종전 `clCreateSubBuffer` 호출 시 alignment 강제 올림(align_to) 로직으로 인해 실제 데이터 시작 위치와 버퍼 오프셋이 어긋남

---

## 2. 근본 원인 분석

### clCreateSubBuffer 정렬 강제 올림 문제
기존 코드에서 서브버퍼 생성 시 아래와 같이 오프셋을 강제로 128바이트 배수로 올림처리:

```cpp
// 기존 코드 (문제 있음)
region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
extra->d = clCreateSubBuffer(extra_orig->data_device, CL_MEM_READ_WRITE, ...);
```

**예시 시나리오:**
- 실제 scales 데이터 위치: offset 2000
- 정렬 후 origin: 2048 (128의 배수)
- 결과: 앞 48바이트 가중치를 건너뛰고 엉뚱한 위치부터 읽음 → **가중치 오염 (garbage weights)**

### 추가 문제: 중첩 서브버퍼 생성 불가
OpenCL 명세상 서브버퍼에서 다시 서브버퍼를 생성하는 것은 불가(Nested subbuffer 금지).
이미 SoA 변환된 텐서에 대해 두 번째 호출이 발생하면 `CL_INVALID_MEM_OBJECT (-38)` 에러.

---

## 3. 해결 조치: cl_create_sub_buffer_fallback 헬퍼 구현

### 구현 위치
`ggml-opencl.cpp` 6897번째 줄 부근에 신규 헬퍼 함수 추가:

```cpp
static cl_mem cl_create_sub_buffer_fallback(
    cl_context context,
    cl_mem parent,
    cl_mem_flags flags,
    cl_buffer_create_type buffer_create_type,
    const void * buffer_create_info,
    cl_int * errcode_ret) {

    cl_mem sub = clCreateSubBuffer(parent, flags, buffer_create_type, buffer_create_info, errcode_ret);
    if (*errcode_ret == CL_SUCCESS) {
        return sub;
    }

    // Fallback: If subbuffer creation fails (e.g. due to alignment or nested subbuffers on Adreno),
    // allocate a new standalone buffer.
    const cl_buffer_region * region = (const cl_buffer_region *) buffer_create_info;
    cl_mem buf = clCreateBuffer(context, flags, region->size, NULL, errcode_ret);
    return buf;
}
```

### 동작 원리
1. **우선** 실제(비정렬) 오프셋으로 서브버퍼 생성 시도
2. **실패 시** (정렬 불일치, 중첩 서브버퍼 등) 독립 `clCreateBuffer` 신규 할당으로 폴백
3. 독립 버퍼는 오프셋=0 → 정렬 문제 원천 해소

---

## 4. 적용된 변경 내역

모든 양자화 타입별 `ggml_backend_opencl_buffer_set_tensor` 내 서브버퍼 생성 로직 수정.
**공통 패턴:** `align_to(..., backend_ctx->alignment)` 제거 + `clCreateSubBuffer` → `cl_create_sub_buffer_fallback`

### 4.1 GGML_TYPE_Q1_0
```cpp
// 변경 전
region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
extra->d = clCreateSubBuffer(extra_orig->data_device, ...);
// ...
region.origin = align_to(previous_origin + size_d, backend_ctx->alignment);
extra->q = clCreateSubBuffer(extra_orig->data_device, ...);

// 변경 후
region.origin = extra_orig->offset + tensor->view_offs + offset;
extra->d = cl_create_sub_buffer_fallback(context, extra_orig->data_device, ...);
// ...
region.origin = previous_origin + size_d;
extra->q = cl_create_sub_buffer_fallback(context, extra_orig->data_device, ...);
```

### 4.2 GGML_TYPE_Q4_0
- `extra->d` (scales), `extra->q` (quants) 동일 패턴 적용

### 4.3 GGML_TYPE_Q4_1
- `extra->d` (scales), `extra->m` (mins), `extra->q` (quants) 동일 패턴 적용

### 4.4 GGML_TYPE_Q5_0
- `extra->d` (scales), `extra->qh` (upper bits), `extra->qs` (lower 4bits) 동일 패턴 적용

### 4.5 GGML_TYPE_Q5_1
- `extra->d`, `extra->m`, `extra->qh`, `extra->qs` 동일 패턴 적용

### 4.6 GGML_TYPE_MXFP4
- `extra->e` (exponents), `extra->q` (quants) 동일 패턴 적용

### 4.7 GGML_TYPE_IQ4_NL
- `extra->d` (scales), `extra->q` (quants) 동일 패턴 적용

### 4.8 GGML_TYPE_Q8_0
- (기존에 별도 guard 로직 있음, fallback 보강)

### 4.9 GGML_TYPE_Q4_K
- `extra->d`, `extra->dm`, `extra->s`, `extra->q` 동일 패턴 적용

### 4.10 GGML_TYPE_Q5_K
- `extra->d`, `extra->dm`, `extra->s`, `extra->q`, `extra->qh` 동일 패턴 적용

### 4.11 GGML_TYPE_Q6_K (MoE 경로 및 일반 경로 모두)
- MoE 경로: `extra->ql`, `extra->qh`, `extra->s`, `extra->d`
- 일반 경로: `extra->ql`, `extra->qh`, `extra->s`, `extra->d`
- 모두 동일 패턴 적용

---

## 5. 빌드 결과

```
ninja 빌드 성공
[3/3] Linking CXX shared module bin/libggml-opencl.so
```

컴파일 에러 없음, 경고 3개(기존 발생 경고)만 남음.

---

## 6. 추가 발견된 문제 (해결됨)

### 커널 컴파일 에러 (err=-6)
llama-server 실행 시 gemma-4 모델 로드 중 아래 에러 발생:
```
0.03.610.966 E ggml_opencl: kernel compile error (err=-6):

Pass
```

- `err=-6` = `CL_OUT_OF_HOST_MEMORY`
- 첫 번째 OpenCL 커널 로딩(`load_cl_kernels`) 시점에 메모리 부족으로 컴파일 실패
- **원인 1**: 약 50개의 OpenCL 프로그램이 `backend_ctx->program_*` 멤버 변수에 저장된 채로 `clReleaseProgram`이 호출되지 않아 OpenCL 드라이버 힙 메모리가 점차 고갈됨
- **원인 2**: Adreno GPU 드라이버에서 `-cl-fast-relaxed-math -cl-unsafe-math-optimizations` 플래그가 셰이더 컴파일러 힙 사용량을 급증시킴

### 해결 조치

**1. clReleaseProgram 일괄 해제 (2026-07-02)**
`load_cl_kernels` 함수 종료 직전 모든 `backend_ctx->program_*` 프로그램 객체를 순회하며 `clReleaseProgram` 호출 후 nullptr 설정.
- OpenCL 명세: "The program object associated with the kernel object can be released after the kernel object is created."
- 약 60개 프로그램 객체를 배열에 담아 일괄 해제
- nullptr-safe 처리로 조건부/미할당 프로그램 대응

```cpp
cl_program * programs[] = {
    &backend_ctx->program_add,
    &backend_ctx->program_cvt,
    // ... (약 60개)
};
for (size_t i = 0; i < sizeof(programs)/sizeof(programs[0]); i++) {
    if (*programs[i]) {
        clReleaseProgram(*programs[i]);
        *programs[i] = nullptr;
    }
}
```

**2. Adreno compile_opts 최적화 (2026-07-02)**
`load_cl_kernels` 및 `load_cl_kernels_argsort` 내 compile_opts에서 Adreno GPU 감지 시 `-cl-unsafe-math-optimizations`와 `-cl-fast-relaxed-math` 플래그를 제거:
```cpp
if (backend_ctx->gpu_family == GPU_FAMILY::ADRENO) {
    compile_opts.erase(compile_opts.find(" -cl-unsafe-math-optimizations"), ...);
    compile_opts.erase(compile_opts.find(" -cl-fast-relaxed-math"), ...);
}
```

---

## 7. 완료된 작업 및 과거 미완료 사항

### 7.1 가중치 틀어짐 추가 검증 (완료)
- `get_tensor` 경로의 복원 로직 오프셋 일관성 검증 완료
  - Q4_0 AoS 경로: `extra_aos->offset + tensor->view_offs + offset` — `set_tensor`와 동일한 계산식
  - Q4_0 SoA 경로: restore kernel 출력 버퍼 → `offset` 파라미터 직접 사용 (올바름)
  - Q1_0, Q4_1, Q5_0, Q5_1 등: 모두 restore kernel → `offset` 직접 사용 패턴
- `set_tensor` 내 서브버퍼 생성: `region.origin = extra_orig->offset + tensor->view_offs + offset` — 정렬 강제 올림 제거됨
- 결론: **get_tensor / set_tensor 간 오프셋 불일치 없음**

### 7.2 커널 컴파일 에러 해결 (완료 — Section 6 참조)
- `clReleaseProgram` 일괄 해제: ~60개 프로그램 객체 메모리 해제
- Adreno compile_opts: `-cl-unsafe-math-optimizations -cl-fast-relaxed-math` 조건부 제거

### 7.3 추가 확인 필요한 파일
- `ggml/src/ggml-opencl/kernels/` 내 모든 `.cl` 파일의 오프셋 처리
- `get_tensor` 함수 전체 경로 (set_tensor와 대칭 여부 확인)

---

## 8. 참고 정보

### 테스트에 사용된 모델
- `unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL` (약 1.5GB)
- `unsloth/gemma-4-E2B-it-GGUF:Q4_0`
- `ggml-org/Qwen3-1.7B-GGUF:Q4_K_M`

### 서버 실행 명령
```bash
export LD_LIBRARY_PATH=/data/data/com.termux/files/usr/lib:/vendor/lib64:/system/lib64
export LD_PRELOAD=/vendor/lib64/libOpenCL.so
./llama-server -m <model.gguf> -t 4 -ngl 20 -fa on --no-mmap --host 0.0.0.0 --port 56810
```

### 이전 수정 이력 (별도 문서 참조)
`opencl_subgroup_shuffle_fix.md` - Flash Attention subgroup shuffle 및 workgroup size 관련 수정 내역

---

*이 문서는 2026-07-02 07:36 KST 기준으로 작업 중단 시점의 상태를 기록한 것입니다.*
