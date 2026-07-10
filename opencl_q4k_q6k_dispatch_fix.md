# Q4_K / Q6_K Adreno Dispatch Mismatch — Root Cause and Fix

작성일: 2026-07-06
브랜치: fix_opencl_v2x_adreno
관련 파일: `ggml/src/ggml-opencl/ggml-opencl.cpp`

이 문서는 "일부 양자화 구성에서 엉뚱한 답이 나온다"는 증상의 실제 원인 중 하나를
실기기(Snapdragon 888 / Adreno 660, SM-F926N, Termux)에서 재현 및 검증하고 수정한
내역을 기록한다. 이전 문서(`opencl_weight_corruption_fix_progress.md`,
`opencl_weight_corruption_fix_changes.md`, `opencl_subgroup_shuffle_fix.md`)에서
다룬 subbuffer/alignment 이슈와는 별개의, 커밋 `9e56db4dc`에서 남은 디버그 코드로
인한 문제다.

## 1. 증상 재현 방법

기존 문서들은 실제 모델(`gemma-4-E2B-it-qat-UD-Q4_K_XL`)로 CPU/GPU 출력을 비교했지만,
이 모델은 실제로는 거의 전부 **Q4_0** 텐서만 사용하고 있어 (`per_layer_*`, `ffn_*`,
`attn_*` 전부 type=2/Q4_0) Q4_K/Q6_K 경로를 전혀 실행하지 않는다. 그래서 겉으로는
정상 동작하는 것처럼 보였다.

실제 원인은 `test-backend-ops`로 **실제 모델 크기(512 이상)의 Q4_K/Q6_K MUL_MAT**를
돌려야 드러난다:

```bash
export LD_LIBRARY_PATH=.../lib:/vendor/lib64:/system/lib64
export LD_PRELOAD=/vendor/lib64/libOpenCL.so
./bin/test-backend-ops -b GPUOpenCL -o MUL_MAT -p "m=1024"
```

수정 전 결과:
```
[MUL_MAT] ERR = 1.878432336 > 0.000500000  MUL_MAT(type_a=q4_K, m=1024,n=1,k=1024): FAIL
[MUL_MAT] ERR = 1.952852044 > 0.000500000  MUL_MAT(type_a=q4_K, m=1024,n=32,k=1024): FAIL
  MUL_MAT(type_a=q5_K, ...): OK
[MUL_MAT] ERR = 1.869693633 > 0.000500000  MUL_MAT(type_a=q6_K, m=1024,n=1,k=1024): FAIL
[MUL_MAT] ERR = 1.958426899 > 0.000500000  MUL_MAT(type_a=q6_K, m=1024,n=32,k=1024): FAIL
```

`m,k < 512`인 작은 텐서(테스트 기본 스위트가 쓰는 크기)에서는 전부 통과하기 때문에
`test-backend-ops`의 기본 케이스만으로는 이 버그를 잡을 수 없다. 반드시 Adreno의
`use_adreno_kernels()` 임계값(ne0, ne1 >= 512, 구형 컴파일러는 >=128)을 넘는 크기로
테스트해야 한다.

## 2. 원인 1: Q4_K — convert와 dispatch의 불일치 (확정, 결정론적 버그)

커밋 `9e56db4dc` ("opencl : Adreno fixes: ... Q4_K noshuffle disabled for Q4_K_XL
diagnostic")에서 디버그 목적으로 `ggml_cl_mul_mat` 내 Adreno 전용 dispatch를
비활성화했다:

```cpp
// q4_k x fp32 -- DEBUG: disabled temporarily to test
if (false && src0t == GGML_TYPE_Q4_K && src1t == GGML_TYPE_F32 && !use_flat_gemv_for_large_m_q4_K(src0)) {
    ggml_cl_mul_mat_q4_k_f32_adreno(backend, src0, src1, dst);
    return;
}
```

문제는 **weight를 SoA로 변환하는 `set_tensor` 쪽은 그대로 두었다는 것**이다.
`ggml_backend_opencl_buffer_set_tensor()`의 Q4_K 변환부는 여전히 조건 없이
`kernel_convert_block_q4_K_noshuffle`을 사용한다 (임계값 이상 텐서에 한해):

```cpp
if (use_adreno_kernels(backend_ctx, tensor) && !use_flat_gemv_for_large_m_q4_K(tensor)) {
    kernel = backend_ctx->kernel_convert_block_q4_K_noshuffle;   // 그대로 활성
}
```

`kernel_convert_block_q4_K_noshuffle`(`cvt.cl`)은 원본 AoS 바이트를 그대로 복사하는
것이 아니라 니블 비트를 재배치(bit-shuffle)해서 저장한다 — Adreno 전용
`kernel_gemv_noshuffle_q4_k_f32` / `kernel_gemm_noshuffle_q4_k_f32`가 런타임에 셔플
없이 읽을 수 있도록 사전 변환하는 용도다. 반면 dispatch가 꺼져 있으니 실제 연산은
범용 `kernel_mul_mv_q4_K_f32_flat`(원본 비트 순서를 기대)로 넘어간다.

**결과: 가중치는 A 포맷으로 저장되고, 커널은 B 포맷으로 해석 → 임계값(512) 이상인
모든 Q4_K 텐서에서 결정론적으로 틀린 결과.** (test-backend-ops로 30회 이상 반복
재현, 항상 동일하게 실패)

Q6_K는 같은 커밋에서 convert(`kernel_convert_block_q6_K_noshuffle` 선택부)와
dispatch를 **둘 다** `false &&`로 껐기 때문에 이런 종류의 불일치는 없었다
(별도 원인은 3절 참조).

### 수정
`ggml_cl_mul_mat()`의 Q4_K dispatch에서 `false &&` 제거, convert 로직과 다시
일치시킴 (16747번째 줄 부근):

```cpp
// q4_k x fp32
if (src0t == GGML_TYPE_Q4_K && src1t == GGML_TYPE_F32 && !use_flat_gemv_for_large_m_q4_K(src0)) {
    ggml_cl_mul_mat_q4_k_f32_adreno(backend, src0, src1, dst);
    return;
}
```

수정 후 `m=1024,n={1,32},k=1024` Q4_K 테스트 모두 통과.

## 3. 원인 2: Q6_K — flat kernel의 다중 workgroup 이슈 (재현 불안정, 별도 조치)

같은 커밋에서 `mul_mv_q6_k_f32_flat.cl`의 리덕션을 `sub_group_reduce_add()`에서
로컬 메모리 기반 수동 트리 리덕션으로 교체했다("Adreno GPU driver subgroup
reduction bugs를 우회하기 위함"이라는 주석 포함). 검증 결과:

- **원래(`sub_group_reduce_add`) 코드로 되돌리면 `m=16`(테스트 기본 크기)조차
  실패한다** → subgroup reduce 자체가 이 드라이버에서 정말 깨져 있다는 주석의
  주장은 사실로 확인됨.
- 로컬 메모리 리덕션(현재 코드)은 작은 크기에서는 항상 통과하지만, `m=1024`
  (workgroup 32개)에서 **처음 한 번** 큰 오차(ERR≈1.9)로 실패한 뒤, 동일 빌드로
  같은 케이스를 20회 이상 반복 실행했을 때는 단 한 번도 재현되지 않았다. 즉
  결정론적 로직 버그라기보다 Adreno 드라이버 쪽 타이밍/워밍업성 문제로 보인다
  (이 프로젝트 히스토리에 이런 종류의 Adreno GPU 드라이버 불안정성으로 인한
  revert 커밋이 다수 존재 — `fa40c9568`, `b50649b24`, `49beb5060` 등 참고).

원인을 100% 확정하지 못했으므로, 안전한 완화책으로 **Q6_K도 Q4_K와 동일하게
Adreno 전용 noshuffle GEMV/GEMM 경로를 다시 활성화**했다 (convert 2곳 + dispatch
1곳, 모두 `use_flat_gemv_for_large_m_q6_K()` 임계값 이상에서만 적용):

```cpp
if (use_adreno_kernels(backend_ctx, tensor) && !use_flat_gemv_for_large_m_q6_K(tensor)) {
    kernel = backend_ctx->kernel_convert_block_q6_K_noshuffle;   // set_tensor, 2곳
}
...
if (src0t == GGML_TYPE_Q6_K && src1t == GGML_TYPE_F32 && !use_flat_gemv_for_large_m_q6_K(src0)) {
    ggml_cl_mul_mat_q6_K_f32_adreno(backend, src0, src1, dst);   // dispatch
    return;
}
```

이렇게 하면 `ne0, ne1 >= 512`인 (실제 모델에서 가장 위험한 큰) Q6_K 텐서는 검증된
Adreno 전용 커널을 타고, `mul_mv_q6_k_f32_flat`(불안정 가능성이 남아있는 generic
경로)은 더 이상 사용되지 않는다.

**남은 리스크**: `32 < ne1 < 512` (즉 여러 workgroup이 필요하지만 Adreno 임계값
미만) 크기의 Q6_K 텐서는 여전히 `mul_mv_q6_k_f32_flat`을 탄다. 이 크기대에서
반복 테스트(`m=64,128,256,512,1024`, 각 10회+)는 전부 통과했지만, 완전히
결정론적으로 안전하다고 보증할 수는 없다. 실기기에서 Q6_K 비중이 큰 모델을
장시간/반복 추론하면서 이상 징후(가끔 튀는 이상한 토큰)가 없는지 지켜볼 필요가
있다.

## 4. 검증

`ggml/src/ggml-opencl/ggml-opencl.cpp` 수정 후 (mem align 관련 subbuffer 로직은
건드리지 않음, 순수하게 `false &&` 3곳 제거):

```bash
cd build_opencl/build-android
ninja test-backend-ops
export LD_LIBRARY_PATH=.../lib:/vendor/lib64:/system/lib64
export LD_PRELOAD=/vendor/lib64/libOpenCL.so
./bin/test-backend-ops -b GPUOpenCL -o MUL_MAT
# 919/919 tests passed, Backend OpenCL: OK  (3회 반복 실행, 매번 통과)
```

전체 diff는 아래 5줄 × 2(대칭) = 3개 지점, 총 5줄 변경:

```diff
-        if (false && use_adreno_kernels(backend_ctx, tensor) && !use_flat_gemv_for_large_m_q6_K(tensor)) {
+        if (use_adreno_kernels(backend_ctx, tensor) && !use_flat_gemv_for_large_m_q6_K(tensor)) {
   (x2, set_tensor 내 convert 선택부)

-        // q4_k x fp32 -- DEBUG: disabled temporarily to test
-        if (false && src0t == GGML_TYPE_Q4_K && ...) {
+        // q4_k x fp32
+        if (src0t == GGML_TYPE_Q4_K && ...) {

-        if (false && src0t == GGML_TYPE_Q6_K && ...) {
+        if (src0t == GGML_TYPE_Q6_K && ...) {
```

## 5. 다음 사람을 위한 메모

- 이 프로젝트에서 OpenCL/Adreno 코드를 수정할 때 **작은 텐서 크기로만
  테스트하면 이런 종류의 버그를 절대 못 잡는다.** `use_adreno_kernels()`
  임계값(기본 512, 구형 컴파일러 128) 이상 크기로 반드시 확인할 것.
- 실제 다운로드된 테스트 모델의 텐서 타입을 gguf 헤더에서 직접 확인하지 않고
  "Q4_K_XL이라는 이름이니 Q4_K를 쓰겠지"라고 가정하면 안 된다 (unsloth의
  "Dynamic Quant" 네이밍은 실제 quant type과 다를 수 있음).
- `if (false && ...)` 같은 디버그 리브아웃은 커밋 메시지에 "DEBUG"라고 써놨어도
  다음 커밋에서 원복하는 걸 잊기 쉽다. 앞으로는 이런 임시 비활성화를 남길 때
  `// TODO(diagnostic, remove before merge)` 같은 더 눈에 띄는 마커를 쓰는 걸
  권장.
