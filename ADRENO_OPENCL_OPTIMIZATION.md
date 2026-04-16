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
- [ ] `ggml-opencl.cpp` 내 Adreno 전용 초기화 로직 추가
- [ ] Adreno 660 지원 여부 확인 루틴 (`CL_DEVICE_NAME`)
- [ ] FP32 -> FP16 커널 변환 및 성능 검증
- [ ] `matmul` 커널의 Image2D 기반 리팩토링
- [ ] Android 빌드 스크립트 (`CMakeLists.txt`) 최적화 플래그 적용

## 4. 디버깅 및 테스트 로그 (업데이트 예정)
*(여기에 발생하는 오류 로그와 해결 과정을 기록합니다)*
