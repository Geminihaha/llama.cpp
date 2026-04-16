# llama.cpp Adreno 660 (OpenCL 2.0) 최적화 및 수정 가이드

이 문서는 Qualcomm Adreno 660 GPU 환경에서 OpenCL 2.0을 활용하여 `llama.cpp`의 성능을 극대화하고 호환성 문제를 해결하기 위한 기술적 가이드를 담고 있습니다.

## 1. 개발 및 빌드 환경 (Android/Adreno)
*   **Target Device:** Adreno 660 (Snapdragon 888 등)
*   **API Support:** OpenCL 2.0 Full Profile
*   **Library Path:** `/system/vendor/lib64/libOpenCL.so` (기기 내 위치)
*   **Build Toolchain:** Android NDK (CMake 활용)

## 2. 주요 최적화 전략 (Adreno 600 Series 특화)

### A. 메모리 관리 (Unified Memory & SVM)
*   **Shared Virtual Memory (SVM):** OpenCL 2.0의 SVM 기능을 활용하여 CPU와 GPU 간의 명시적 복사(`clEnqueueWriteBuffer`)를 제거합니다.
*   **Host Pointer 활용:** `CL_MEM_ALLOC_HOST_PTR`을 사용하여 Zero-copy 메커니즘을 구현, Adreno의 통합 메모리 아키텍처 이점을 활용합니다.

### B. 데이터 레이아웃 및 하드웨어 가속
*   **Image vs Buffer:** 행렬 연산(`matmul`) 시 Buffer 대신 `image2d_t` (Texture)를 활용하여 Adreno의 텍스처 프로세싱 유닛(TPU) 하드웨어 가속을 유도합니다.
*   **FP16 (Half Precision):** Adreno 660은 FP16 연산 효율이 매우 높습니다. 커널 내부에서 `half` 타입을 적극 활용하고, GGML의 연산 정밀도를 최적화합니다.

### C. 커널 튜닝 (Kernel Optimization)
*   **Work-group Size:** Adreno 아키텍처에 최적화된 로컬 워크그룹 크기(보통 64, 128)를 실험적으로 도출합니다.
*   **Vectorization:** `float4`, `half8` 등의 벡터 타입을 사용하여 ALU 점유율을 높입니다.
*   **Subgroups:** `cl_khr_subgroups` 확장 기능을 사용하여 워프(Warp) 단위의 효율적인 데이터 공유를 구현합니다.

## 3. 구현 및 수정 체크리스트
- [x] `ggml-opencl.cpp` 내 커널 컴파일 옵션에 `-DGGML_OPENCL_USE_ADRENO_KERNELS` 추가
- [x] `rms_norm.cl`: Subgroup 기능 우회 및 RMS 계산 수식 교정
- [x] `mean.cl`: Adreno 6xx용 로컬 메모리 Reduction 구현
- [x] `sum_rows.cl`: Adreno 6xx용 로컬 메모리 Reduction 구현
- [x] `cumsum.cl`: Adreno 6xx용 로컬 메모리 Prefix Sum (Scan) 구현
- [x] `group_norm.cl`: Adreno 6xx용 로컬 메모리 Reduction 구현
- [x] `mul_mv_q4_k_f32.cl`, `mul_mv_f16_f16.cl`, `mul_mv_q8_0_f32.cl`: FP16 연산 최적화
- [x] `gemv_noshuffle_q4_k_f32.cl`: Adreno 전용 FP16/Texture (read_imageh) 최적화 적용
- [x] `ggml-opencl.cpp`: Adreno 최적화 커널 활성화 임계값 하향 (512 -> 64)

## 4. 디버깅 및 테스트 로그

### [2026-04-16] 초기 안정화 및 FP16 최적화
#### 문제 현상
*   **증상:** OpenCL 커널 로딩 중 `[rms_norm]`, `[mean]`, `[sum_rows]`, `[cumsum]`, `[group_norm]` 단계에서 Segmentation Fault 발생.
*   **원인:** Adreno 660 드라이버 컴파일러가 OpenCL 3.0용 Subgroup 확장 기능을 파싱하는 과정에서 크래시 발생.

#### 해결 내역
1.  **호스트 환경 개선:** `-DGGML_OPENCL_USE_ADRENO_KERNELS` 매크로를 커널 컴파일러 옵션에 명시적으로 추가.
2.  **커널 안정화:** `__OPENCL_VERSION__` 기반 분기 로직을 도입하여 Adreno 6xx(OpenCL 2.0)에서는 Subgroup 기능 대신 안전한 `local memory` Reduction/Scan 로직을 사용하도록 수정.
3.  **FP16 가속 적용:** 
    *   `mul_mv` 계열 커널의 내부 연산을 `half` 타입으로 전환하여 하드웨어 가속 유도.
    *   Adreno 전용 `gemv_noshuffle` 커널에 `read_imageh` (FP16 텍스처 읽기) 적용 및 연산 전체를 `half` 정밀도로 최적화.
4.  **성능 범위 확대:** `use_adreno_kernels` 임계값을 64로 낮추어 소형 레이어에서도 Adreno 특화 커널이 작동하도록 조정.
5.  **메모리 최적화 시도:** `CL_MEM_ALLOC_HOST_PTR`을 통한 Zero-copy 기반 마련 (불안정한 Map/Unmap 로직은 안정성을 위해 롤백 후 내부 최적화 유지).

#### 결과
*   모든 커널 로딩 성공 및 Segfault 해결.
*   응답 지연 시간(Latency) 개선 확인.
*   Adreno 660 GPU 가속 엔진 정상 가동 확인.
