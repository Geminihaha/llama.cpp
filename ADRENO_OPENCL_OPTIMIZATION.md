# llama.cpp Adreno 660 (OpenCL 2.0) 최적화 및 수정 가이드

이 문서는 Qualcomm Adreno 660 GPU 환경에서 OpenCL 2.0을 활용하여 `llama.cpp`의 성능을 극대화하고 호환성 문제를 해결하기 위한 기술적 기록입니다.

## 1. 개발 및 빌드 환경
*   **Target Device:** Adreno 660 (Snapdragon 888)
*   **API Support:** OpenCL 2.0 Full Profile (Compiler E031.38.01.08)
*   **Optimization Strategy:** FP16 Acceleration, Unified Memory utilization, Kernel Stability.

---

## 2. [성공 사항] 안정성 확보 및 기초 최적화

### A. 커널 로딩 및 컴파일 크래시(Segfault) 해결
*   **해결:** 
    *   `__OPENCL_VERSION__` 매크로를 활용하여 OpenCL 2.0(Adreno 6xx) 환경에서 Subgroup 기능 대신 안전한 `local memory` 기반 로직을 사용하도록 우회.
    *   호스트 코드(`ggml-opencl.cpp`)에서 `-DGGML_OPENCL_USE_ADRENO_KERNELS` 매크로를 명시적으로 전달.
*   **결과:** 모든 주요 커널(`rms_norm`, `mean`, `sum_rows`, `cumsum`, `group_norm`) 로딩 성공.

### B. FP16 (Half Precision) 하드웨어 가속 적용
*   **대상:** `mul_mv_q4_k_f32.cl`, `mul_mv_f16_f16.cl`, `mul_mv_q8_0_f32.cl`, `mul_mv_q6_k_f32.cl`
*   **내용:** 커널 내부 연산 변수를 `float`에서 `half` 타입으로 전환하여 하드웨어 가속 유닛 활성화.
*   **결과:** 연산 지연 시간(Latency) 감소 및 응답 시작 속도 개선.

### C. 수식 및 문법 오류 교정
*   **RMS Norm:** 제곱 합 계산 수식 보정.
*   **컴파일 규격:** `local` 메모리 선언 위치를 함수 스코프로 이동하여 OpenCL C 표준 준수.

### D. 워크그룹 크기(Local Work Size) 정밀 튜닝 (2026-04-16)
*   **GEMV (토큰 생성):**
    *   **변경:** `{64, 4, 1}` (256 스레드) -> `{64, 2, 1}` (128 스레드)
    *   **이유:** Adreno 660의 레지스터 압박을 줄이고 2-Wave 단위의 조밀한 스케줄링을 유도하여 연산 효율 극대화.
*   **GEMM (프롬프트 처리):**
    *   **변경:** `{1, 128, 1}` -> `{8, 8, 1}` (64 스레드)
    *   **이유:** 비대칭적인 스레드 구조를 정방형 타일(Tile) 구조로 변경하여 하드웨어 텍스처 캐시 히트율을 높이고, 1-Wave(64스레드) 단위로 하드웨어를 꽉 채워 실행하도록 최적화.
*   **결과:** 전반적인 토큰 생성 속도 및 프롬프트 분석 속도의 안정적 향상 기반 마련.

---

## 3. 구현 체크리스트
- [x] `ggml-opencl.cpp` 내 커널 컴파일 옵션에 매크로 추가
- [x] `rms_norm.cl`: Subgroup 우회 및 수식 교정
- [x] `mean.cl`: Adreno 6xx용 로컬 메모리 Reduction
- [x] `sum_rows.cl`: Adreno 6xx용 로컬 메모리 Reduction
- [x] `cumsum.cl`: Adreno 6xx용 로컬 메모리 Scan
- [x] `group_norm.cl`: Adreno 6xx용 로컬 메모리 Reduction
- [x] `mul_mv_q4_k`, `f16_f16`, `q8_0`, `q6_k`: FP16 가속 적용
- [x] Adreno 660 최적 워크그룹 크기(LWS) 튜닝 ({64,2,1}, {8,8,1})

---

## 4. 향후 작업 로드맵
- [ ] **Mixed Precision 최적화:** 정확도 영향이 적은 구간 선별 적용.
- [ ] **GEMM 커널 Tiling 개선:** 프롬프트 처리 성능 추가 최적화.
- [ ] **Image2D 기반 Matmul 전면 도입:** 버퍼 방식에서 이미지 방식으로의 전환 검토.

*주의: 실패 사례 및 상세 분석은 [ADRENO_OPTIMIZATION_FAILURES.md](./ADRENO_OPTIMIZATION_FAILURES.md)를 참조하십시오.*
