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
*   **Host Pointer 활용:** `CL_MEM_USE_HOST_PTR`을 사용하여 Zero-copy 메커니즘을 구현, Adreno의 통합 메모리 아키텍처 이점을 활용합니다.

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

## 4. 디버깅 및 테스트 로그

### 문제 현상
*   **증상:** OpenCL 커널 로딩 중 `[rms_norm]`, `[mean]`, `[sum_rows]`, `[cumsum]`, `[group_norm]` 단계에서 Segmentation Fault 발생.
*   **원인:** Adreno 660 드라이버(OpenCL 2.0, Compiler E031.38.01.08)의 컴파일러가 Subgroup 확장 기능(`sub_group_reduce_add` 등) 및 전용 속성(`qcom_reqd_sub_group_size`)을 파싱하거나 최적화하는 과정에서 내부 오류로 크래시 발생.

### 해결 방법
1.  **호스트 코드 수정 (`ggml-opencl.cpp`):**
    *   커널 컴파일 시 `-DGGML_OPENCL_USE_ADRENO_KERNELS` 매크로를 명시적으로 전달하여 커널 내부에서 Adreno 특화 분기 로직이 작동하도록 함.
2.  **커널 코드 수정 (공통 전략):**
    *   `__OPENCL_VERSION__ >= 300` 조건을 사용하여 OpenCL 3.0 이상을 지원하는 최신 Adreno(7xx, 8xx)에서는 고성능 Subgroup 기능을 유지함.
    *   OpenCL 2.0 환경(Adreno 6xx 등)에서는 컴파일러 크래시를 방지하기 위해 `local memory` 기반의 Reduction/Scan 로직으로 우회 구현.
    *   `qcom_reqd_sub_group_size` 속성이 컴파일러 파싱 에러를 유발하는 경우 조건부로 비활성화.
3.  **수식 교정:**
    *   `rms_norm.cl`에서 제곱 합 계산 시 누락되었던 제곱 연산을 추가하여 연산 정확도 확보.

### 결과
*   모든 OpenCL 커널이 정상적으로 로드됨.
*   Adreno 660 GPU를 활용한 `llama-server` 및 추론 엔진 정상 동작 확인.
