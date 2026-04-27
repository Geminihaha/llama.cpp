# ADRENO_VULKAN_STABILIZATION_REPORT.md

## 1. 개요
*   **목표:** Android(Termux) 환경에서 Adreno 660 GPU의 Vulkan 백엔드를 활용하여 Gemma-4/3n 모델 가속.
*   **현재 상태:** CPU 단독 실행 시 약 12 t/s로 정상 작동 확인. Vulkan 가속 시도 중 수치 오류 및 세그먼테이션 폴트(Segmentation fault) 발생으로 인해 하이브리드 패치 진행.

## 2. 주요 오류 현상 및 원인 분석

| 현상 | 발생 지점 | 원인 추정 |
| :--- | :--- | :--- |
| **수치 오류 (`avg_err`)** | `SCALE` 연산 (node_49) | Adreno 드라이버의 FP16/FP32 정밀도 타협 및 IEEE-754 미준수. |
| **Pipeline Creation Failed** | `Q4_K` 행렬 연산 커널 컴파일 | Adreno 드라이버의 레지스터(Register) 부족 혹은 셰이더 복잡도 한계. |
| **Segmentation Fault** | 프롬프트 처리 중 (70~94%) | Gemma 4/3n의 **Shared KV layers** 및 **ISWA** 로직이 Vulkan Buffer View와 충돌. |
| **ggml_abort** | `initializing slots` 단계 | Vulkan VRAM 할당 및 동기화 과정에서 드라이버 내부 메모리 관리 오류. |

## 3. 시도했던 방법 및 패치 내역

### [방법 1] 런타임 옵션 조정
*   **`-b 1`**: 배치 사이즈를 최소화하여 메모리 압박 완화 시도.
*   **`-fa off`**: 불안정한 Flash Attention 커널 비활성화.
*   **`--no-warmup`**: 초기 수치 검증 단계 건너뛰기.
*   **`-fit off`**: 자동 메모리 피팅 로직 비활성화.

### [방법 2] `ggml-vulkan.cpp` 소스 패치 (연산 우회)
*   **`GGML_OP_SCALE`**: Qualcomm 장치에서 CPU 우회 패치 적용 (수치 오차 방지).
*   **`GGML_OP_FLASH_ATTN_EXT`**: Qualcomm 장치 가속 제외 (안정성).
*   **`K-Quants (Q4_K, Q5_K, Q6_K)`**: Qualcomm에서 CPU 우회 (컴파일 오류 방지).

### [방법 3] `src/llama-model.cpp` 소스 패치 (메모리 우회)
*   **`get_layer_buft_list`**: Adreno GPU 감지 시 레이어 장치를 CPU로 강제 변경.
*   **결과**: KV 캐시가 RAM에 할당되도록 유도하여 슬롯 초기화 시의 세그폴트 차단 시도.

### [방법 4] Vulkan 최하단 할당자 패치
*   **`ggml_backend_vk_buffer_type_alloc_buffer`**: Qualcomm 장치일 경우 Vulkan VRAM 대신 CPU RAM 버퍼를 반환하도록 강제.

## 4. 최종 실패 원인 요약 (왜 Vulkan이 아직 안되는가?)
1.  **Gemma 4 구조적 복잡성**: Per-layer Embedding 및 Shared KV 구조에서 발생하는 복잡한 텐서 참조가 Adreno 드라이버의 버그 유발.
2.  **드라이버 불안정성**: 대규모 연산 그래프 실행 시 드라이버 내부 메모리 오염 발생.
3.  **잔여 GPU 연산**: 캐시와 주요 연산을 우회했음에도 워밍업 단계의 일부 연산이 여전히 GPU에서 터짐.

## 5. 향후 재시도 시 가이드라인
*   **드라이버**: Mesa-Turnip 드라이버(오픈소스)로 교체 권장.
*   **정밀도**: 모든 연산을 FP32로 강제하는 패치 검토.
*   **아키텍처 제어**: `gemma4` 아키텍처 자체를 Qualcomm GPU 가속 대상에서 조건부 제외.
