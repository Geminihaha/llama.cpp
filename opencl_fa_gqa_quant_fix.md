# OpenCL Flash-Attention: GQA + quant-KV garbage output (Adreno)

## 증상
Adreno 660(Snapdragon 888)에서 gemma-3/4 계열 모델을 `-fa on --cache-type-k q8_0
--cache-type-v q8_0` 로 GPU 오프로드(`-ngl` ≥ 1)하면, 생성 텍스트에 엉뚱한
단어/언어가 끼거나 시퀀스가 길어질수록 `**An ** **An **...`, `-n-tly-h...`
같은 garbage로 붕괴함. F16 KV 캐시에서는 정상.

## 근본 원인
OpenCL flash-attention 커널 중 **f32 및 quant(q8_0/q4_0) KV 경로**가 아래 조합에서
잘못된 결과(garbage)를 냄:

- **Grouped-Query Attention** (`n_head_kv < n_head`, 즉 gqa_ratio > 1) **그리고**
- **큰 head 크기** (`dk >= 128`)

`f16` / `f32+f16 mixed` KV 경로만 이 조합에서 정상. 두 경로는 KV를 local/private
메모리에 half로 담아 풋프린트가 절반이라, dk=256의 큰 private array(`q_priv[64]`,
`o_acc[64]` 등)로 인한 Adreno 컴파일러 문제(register/scratch 스필 미스컴파일)의
임계값 아래로 유지되는 것으로 추정.

### 재현 (test-backend-ops)
`tests/test-backend-ops.cpp` 에 gemma SWA 레이어 구성을 추가해 확인:

| 구성 | 결과(수정 전) |
|---|---|
| dk=256, dv=256, GQA 8:1, q8_0 KV | **ERR ≈ 2.0 (FAIL)** |
| dk=256, dv=256, GQA 8:1, q4_0 KV | **ERR ≈ 2.0 (FAIL)** |
| dk=256, dv=256, GQA 8:1, **f16 KV** | **OK** |
| dk=256, dv=256, **GQA 없음**(nr2=1), q8_0 KV | OK |
| dk=128, GQA 8:1, q8_0 KV | FAIL |
| dk=64, GQA, q8_0 KV | OK |

판별 테스트로 트리거가 **gqa_ratio > 1** 임을 확정(ratio=1·head 8개는 OK,
ratio=2는 FAIL → 워크그룹 수가 아니라 GQA 비율이 원인). 또한 GPU dequant→f32
폴백 경로도 같은 f32 커널을 써서 동일하게 실패함을 확인.

기존 upstream test-backend-ops는 quant KV를 hsk=64/72 에서만 테스트(코드 9146행)해
이 대형-head + GQA 조합이 커버되지 않아 그동안 잡히지 않았음.

## 수정
`ggml_opencl_supports_op()` (GGML_OP_FLASH_ATTN_EXT) 에서 깨지는 조합을 거부해
**CPU 폴백** 시킴. f16 KV 경로는 그대로 GPU에서 실행:

```cpp
const bool gqa_large_head = k->ne[2] < q->ne[2] && dk >= 128;
const bool kv_is_f16_path = k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16;
if (gqa_large_head && !kv_is_f16_path) {
    return false;   // -> CPU fallback (correct)
}
```

- gemma-3/4 dk=512 full-attn 레이어는 원래 supported_dims 에 없어 이미 CPU 폴백.
- 이 수정으로 dk=256 SWA 레이어(q8_0/q4_0 KV)도 CPU 폴백 → 정확한 출력.
- 비-GQA quant KV, dk<128 quant KV, 모든 f16 KV 는 영향 없음(그대로 GPU).

## 검증 (실제 모델, gemma-4-E2B-it-qat Q4_K_XL)
- 수정 전: `The capital of France is **An ** **An ** **An **...` (garbage)
- 수정 후 (q8_0 KV, `-ngl 99`): `The capital of France is Paris.` / 200토큰 단편
  소설 완전 일관 출력, 붕괴 없음.

## 성능 참고 (Adreno 660)
정확성은 두 KV 타입 모두 확보됐으나, 이 기기에서 속도 차이가 큼:

| KV 캐시 | eval | prompt |
|---|---|---|
| q8_0 (dk=256 SWA가 CPU) | 2.47 tok/s | 16.6 tok/s |
| **f16 (dk=256 SWA가 GPU)** | **5.55 tok/s** | **28.9 tok/s** |

→ **이 기기에서는 `--cache-type-k f16 --cache-type-v f16` 사용을 권장** (약 2.2배 빠름).
`build_opencl/build-android/bin/lcs.hf.sh` 의 `--cache-type-k/v q8_0` 를 `f16` 로
바꾸면 됨. q8_0 KV 는 정확하지만 attention 이 CPU 로 내려가 느림.

## 남은 이슈 (별개, 이번 수정 범위 아님)
test-backend-ops FLASH_ATTN_EXT 에서 `hsk=40 + nr3=3`(dim-3 batch broadcast) +
mask + prefill(nb=75) 케이스 20개가 여전히 실패. head_dim=40 은 gemma-4 등에서
쓰이지 않으며 단일 시퀀스 추론과 무관. 추후 조사 대상.
