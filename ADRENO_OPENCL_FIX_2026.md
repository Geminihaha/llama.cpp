# Adreno GPU OpenCL 빌드 및 런타임 오류 수정 보고서 (2026-06-18)

이 문서는 Qualcomm Adreno GPU 환경(Termux Android)에서 `llama.cpp` 최신 소스를 OpenCL 가속을 활성화하여 실행할 때 발생했던 라이브러리 링크 에러, GPU 디바이스 탐지 실패, 그리고 커널 컴파일 세그멘테이션 폴트(Segfault) 오류의 원인 분석과 해결 방법을 정리한 기록입니다.

---

## 1. 개요 및 발생한 문제

최근 `llama.cpp` 소스를 업데이트한 이후 OpenCL 백엔드 빌드 후 실행 시 다음과 같은 세 단계의 오류로 인해 GPU 가속 실행이 불가능했습니다.

1. **라이브러리 링크 오류 (런타임 로딩 실패)**
   ```
   CANNOT LINK EXECUTABLE "./llama-server": cannot locate symbol "_ZTVNSt6__ndk117bad_function_callE" referenced by ".../libllama-server-impl.so"
   ```
2. **GPU 디바이스 탐지 실패**
   - 링크 에러를 해결한 후, 실행 파일이 시스템 GPU(Qualcomm Adreno)를 인식하지 못하고 CPU 백엔드로만 동작하는 현상 발생.
3. **GPU 컴파일러 세그멘테이션 폴트 (Segfault)**
   - GPU 백엔드가 활성화되어 모델을 로드하고 커널을 컴파일하는 도중 세그멘테이션 폴트와 함께 프로세스가 강제 종료되는 현상 발생.

---

## 2. 원인 분석 및 해결 조치

### A. 라이브러리 링크 오류 해결
* **원인:** Termux 환경에서는 자체 컴파일러와 링커가 로컬 C++ 표준 라이브러리(`libc++_shared.so`)를 기준으로 빌드합니다. 그러나 실행 시 `LD_LIBRARY_PATH` 경로 설정 문제로 인해 Android 시스템 영역(`/vendor/lib64` 또는 `/system/lib64`)에 있는 호환되지 않는 버전의 표준 라이브러리가 먼저 로드되면서 심볼 링크 참조 오류가 발생했습니다.
* **해결:** 실행 스크립트([lcs.hf.sh](file:///data/data/com.termux/files/home/llama.cpp/build_opencl/build-android/bin/lcs.hf.sh)) 내에서 `LD_LIBRARY_PATH` 환경 변수 설정 시 Termux의 라이브러리 경로(`/data/data/com.termux/files/usr/lib`)를 가장 처음에 오도록 우선순위를 명시적으로 지정했습니다.
  ```bash
  export LD_LIBRARY_PATH=/data/data/com.termux/files/usr/lib:/vendor/lib64:/system/lib64
  ```

### B. GPU 탐지 오류 해결 (ICD 드라이버 우회)
* **원인:** Termux 패키지 매니저로 빌드된 OpenCL 라이브러리는 `ocl-icd`라는 ICD 로더(wrapper)를 사용합니다. 이 로더가 Qualcomm Adreno의 OpenCL 확장 규격(`clIcdGetPlatformIDsKHR`)을 호출할 때 `-30 (CL_INVALID_VALUE)` 에러를 반환하며 플랫폼을 찾지 못하는 호환성 버그가 존재합니다.
* **해결:** 시스템에 탑재된 Adreno 벤더 고유의 OpenCL 라이브러리인 `/vendor/lib64/libOpenCL.so`를 직접 프리로드하도록 설정하여, 호환되지 않는 ICD 로더를 바이패스시켰습니다.
  ```bash
  export LD_PRELOAD=/vendor/lib64/libOpenCL.so
  ```

### C. GPU 컴파일러 세그멘테이션 폴트 해결 (커널 수정)
* **원인:** Adreno GPU 드라이버 내장 컴파일러(`/vendor/lib64/libllvm-qcom.so`)는 OpenCL 커널 최적화 단계(`QGPUPeepholeOptimizer::lowerPseudoSubgroupArithOpWithAdvSubgroupFeature`)에서 Subgroup 산술 연산 매크로(`sub_group_reduce_add`)를 처리할 때 컴파일러 단에서 내부 세그멘테이션 폴트를 일으키며 크래시가 발생합니다.
* **대상 파일:** [mul_mv_f16_f32_l4.cl](file:///data/data/com.termux/files/home/llama.cpp/ggml/src/ggml-opencl/kernels/mul_mv_f16_f32_l4.cl) 의 `kernel_mul_mat_f16_f32_l4` 및 `kernel_mul_mat_f16_f32_l4_dr` 커널.
* **해결 조치:**
  1. **로컬 메모리 리덕션 Fallback 추가:** Adreno 가속 매크로(`GGML_OPENCL_USE_ADRENO_KERNELS`)가 활성화된 경우, Subgroup 가속 대신 스레드 그룹 간 공유 로컬 메모리(`l_sum`)를 활용하는 리덕션 방식을 사용하도록 조건부 컴파일 분기(`#if / #else`)를 작성하였습니다.
  2. **OpenCL C 표준 스코프 준수:** OpenCL C 명세상 공유 메모리(`local float l_sum[1024]`) 지시어는 중첩 루프나 분기문 내부가 아닌 커널 함수의 최상단(Function Scope)에만 정의되어야 합니다. 초기 수정 시 루프 내부에 잘못 선언되어 컴파일 에러(`__local can only appear in __kernel functions at function scope`)가 발생했던 부분을 함수 도입부로 옮겨 빌드를 완벽히 성공시켰습니다.

---

## 3. 수정 전후 코드 비교

### [mul_mv_f16_f32_l4.cl](file:///data/data/com.termux/files/home/llama.cpp/ggml/src/ggml-opencl/kernels/mul_mv_f16_f32_l4.cl) 수정본 반영사항

#### 1) `kernel_mul_mat_f16_f32_l4` 함수 스코프 선언 및 Fallback
```cl
kernel void kernel_mul_mat_f16_f32_l4(
    ...
) {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    local float l_sum[1024]; // 함수 최상단에 local 변수 선언
#endif
    ...
    for (int r1 = 0; r1 < nrows; ++r1) {
        ...
#if defined(cl_khr_subgroups) && (__OPENCL_VERSION__ >= 300 || !defined(GGML_OPENCL_USE_ADRENO_KERNELS))
        float all_sum = sub_group_reduce_add(sumf);
        if (get_sub_group_local_id() == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
#else
        // Adreno GPU 전용 컴파일러 크래시 우회용 로컬 메모리 리덕션
        l_sum[get_local_id(0)] = sumf;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = get_local_size(0) / 2; i > 0; i /= 2) {
            if (get_local_id(0) < i) {
                l_sum[get_local_id(0)] += l_sum[get_local_id(0) + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (get_local_id(0) == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = l_sum[0];
        }
#endif
    }
}
```

---

## 4. 최종 결과 확인

수정 코드를 컴파일하고 실행 스크립트로 동작을 점검한 결과, 오류가 완전히 해소되고 정상 기동됨을 확인하였습니다.

* **OpenCL 가속 백엔드 빌드:** 성공
* **Qualcomm Adreno GPU 탐지:** 성공 (`GPUOpenCL: QUALCOMM Adreno(TM)`)
* **커널 컴파일 & 모델 가속 로딩:** 성공
* **서버 동작:** 성공 (`llama_server: server is listening on http://0.0.0.0:56810`)
