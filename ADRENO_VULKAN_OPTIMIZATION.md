# llama.cpp Adreno (Vulkan) 최적화 및 수정 가이드

이 문서는 Qualcomm Adreno GPU 환경에서 Vulkan 백엔드를 활용하여 `llama.cpp`의 안정성을 확보하고 성능을 최적화하기 위한 기술적 기록입니다.

---

## 1. 개발 및 빌드 환경
*   **Target Device:** Adreno 660 (Snapdragon 888) 및 최신 Snapdragon 시리즈
*   **API Support:** Vulkan 1.1+ (Qualcomm Proprietary Driver / Mesa Turnip)
*   **Key Issues:** Adreno 드라이버의 호스트 메모리 공유 기능(UVA) 불안정성 해결.

---

## 2. [성공 사항] 안정성 확보 및 기초 수정

### A. 초기화 및 로딩 시 세그먼트 폴트(Segmentation Fault) 해결
*   **증상:** `-fit on`(기본값) 상태에서 메모리 계산 시 또는 모델 로드(`load_tensors`) 중 즉시 크래시 발생.
*   **원인:** Adreno 드라이버가 `VK_EXT_external_memory_host` 확장을 지원한다고 보고하지만, 실제 호스트 메모리 임포트/익스포트 과정에서 메모리 정렬(Alignment) 문제로 세그먼트 폴트 유발.
*   **해결:** `ggml-vulkan.cpp` 내에서 `VK_VENDOR_ID_QUALCOMM` 감지 시 해당 확장을 강제로 비활성화하도록 수정.
*   **결과:** Adreno GPU에서 `-ngl` 옵션을 사용한 모델 로딩 및 초기화 성공.

### B. 비동기 연산 및 UMA 구조 최적화
*   **비동기 연산(Async) 비활성화:** Adreno 드라이버는 다중 큐(Multi-queue) 사용 시 레이스 컨디션으로 인한 크래시가 빈번함. Qualcomm 장치 감지 시 `support_async`를 강제로 `false`로 설정.
*   **UMA(Unified Memory Architecture) 강제 적용:** Adreno GPU가 통합 메모리 구조임을 명시하여, 호스트 가시 메모리 활용도를 높이고 불필요한 메모리 복사 오버헤드를 방지함.

### C. Shader Device Address 기능 비활성화 (보안 패치)
*   **증상:** 모델 로드(`load_tensors`) 도중 또는 `-ngl 0` 설정임에도 불구하고 세그먼트 폴트 발생.
*   **원인:** Adreno 드라이버가 `buffer_device_address` 기능을 지원한다고 보고하지만, 실제 포인터 주소를 계산하거나 텐서 데이터를 가상 매핑할 때 드라이버 내부 메모리 오염 발생.
*   **해결:** `ggml-vulkan.cpp`에서 Qualcomm 장치 감지 시 `buffer_device_address`를 강제로 `false`로 설정하여 안정성 확보.

---

## 3. 구현 체크리스트
- [x] `ggml-vulkan.cpp`: Qualcomm Vendor ID (`0x5143`) 감지 로직 추가
- [x] `ggml-vulkan.cpp`: Adreno GPU에서 `external_memory_host` 확장 비활성화 패치 적용
- [x] `ggml-vulkan.cpp`: Adreno GPU에서 비동기 연산 비활성화 및 UMA 강제 적용
- [x] `ggml-vulkan.cpp`: Adreno GPU에서 `buffer_device_address` 비활성화 (Segmentation Fault 방지)
- [ ] `vulkan-shaders`: Adreno 6xx용 `subgroup` 연산 fallback (Shared Memory 활용) 구현
- [ ] `ggml-vulkan.cpp`: Adreno 맞춤형 Local Work Size (LWS) 자동 튜닝 로직

---

## 4. 권장 실행 옵션 (안드로이드/Adreno 환경)
성공적인 실행을 위해 다음 옵션 조합을 권장합니다:
*   **`-fit off`**: 초기 메모리 자동 계산 과정에서의 드라이버 충돌 방지.
*   **`-c 2048` (또는 이하)**: 모바일 메모리 제약을 고려한 컨텍스트 크기 조정.
*   **`--no-mmap` 제거**: 안드로이드에서는 시스템 기본 메모리 관리를 사용하는 것이 더 안정적임.
*   **`-ngl [레이어수]`**: 레이어를 점진적으로 늘려가며 GPU 가용 메모리 확인.

---

## 5. 향후 작업 로드맵
- [ ] **Shader Optimization:** Adreno의 Wave size(64/128)에 최적화된 연산 로직 이식.
- [ ] **Mixed Precision:** FP16 가속 유닛을 풀 가동하기 위한 SPIR-V 셰이더 검토.
- [ ] **Mesa Turnip 지원 강화:** 오픈소스 드라이버와의 호환성 및 성능 비교 벤치마크.

---
*주의: OpenCL 관련 최적화 기록은 [ADRENO_OPENCL_OPTIMIZATION.md](./ADRENO_OPENCL_OPTIMIZATION.md)를 참조하십시오.*
