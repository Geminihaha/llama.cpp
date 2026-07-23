# llama.cpp CPU-Only 추론 처리 순서 및 기술·수학적 이론

이 문서는 `llama.cpp`를 CPU 전용(CPU-Only) 환경에서 빌드하고 실행할 때 작동하는 **전체 처리 순서(Processing Sequence)**와 각 단계에 적용된 **컴퓨터 공학적 기술 및 수학적 이론**을 순차적으로 정리한 문서입니다.

---

## 1. 개요 (Overview)

`llama.cpp`는 GPU가 없는 일반 CPU 환경에서 대규모 언어 모델(LLM)을 고속으로 추론할 수 있도록 설계된 C/C++ 기반 프레임워크입니다.
CPU 추론 시 주요 병목인 **메모리 대역폭(Memory Bandwidth)** 한계를 극복하고, **CPU SIMD 벡터 연산기**의 성능을 최대로 끌어올리기 위해 로딩부터 토큰 생성까지 모든 파이프라인이 정밀하게 최적화되어 있습니다.

---

## 2. 순차적 전체 처리 파이프라인 (Execution Workflow)

```
[1. 모델 로딩]         GGUF 파일 mmap 매핑 및 가중치 메타데이터 파싱
       │
[2. 그래프 구축]       ggml_cgraph 생성 및 ggml-alloc 기반 정적 메모리 버퍼 할당
       │
[3. 프롬프트 입력]     Tokenization 및 Input Embedding Lookup
       │
[4. Prefill 단계]      입력 프롬프트 병렬 처리 (GEMM 연산 중심)
       │              ├── RMSNorm (Single-pass Normalization)
       │              ├── Q, K, V Projection (Quantized Matrix Mult)
       │              ├── RoPE 위치 인코딩 (Rotary Complex Rotation)
       │              ├── FlashAttention (Online Softmax Tiling)
       │              └── FFN / SwiGLU 연산 (Vectorized SiLU)
       │
[5. Decode 단계]       토큰 단위 순차적 자동거듭생성 (GEMV 연산 중심)
       │              ├── 동적 활성화 양자화 (On-the-fly Q8_0/Q8_K)
       │              ├── SIMD 점곱 연산 (AVX-512/VNNI, ARM DotProd 등)
       │              └── KV Cache 갱신 및 재활용
       │
[6. 출력 및 샘플링]    LM Head Projection → Logits → Temperature/Top-P/Top-K → Token 출력
```

---

## 3. 단계별 상세 처리 순서 및 기술·수학적 이론

### 단계 1: 모델 파일 로딩 및 메모리 매핑 (Model Loading & Memory Mapping)

#### 처리 순서
1. `GGUF` 포맷 모델 파일의 헤더, 텐서 메타데이터(양자화 타입, Shape, Offset) 파싱.
2. `mmap()` 시스템 콜을 호출하여 가중치 파일을 가상 메모리 공간에 직접 매핑.

#### 적용 기술 및 컴퓨터 공학 이론
* **Zero-Copy Memory Mapping (`mmap`)**:
  * 디스크 데이터를 RAM으로 전체 복사(Copy)하는 과정 없이, OS 커널의 Page Cache에 가상 메모리 주소를 즉시 바인딩합니다.
  * 추론 중 실제 액세스되는 가중치 블록만 운영체제가 Page In 시키므로, 모델 로딩 시간이 0초에 가깝고 복수 프로세스 구동 시 RAM을 공유합니다.

---

### 단계 2: 연산 그래프 생성 및 정적 메모리 할당 (Graph Construction & Memory Allocation)

#### 처리 순서
1. 입력 토큰 길이 및 모델 아키텍처(Layer 수, Head 수, Hidden Dim)를 기반으로 `ggml_cgraph` 연산 그래프 노드 구성.
2. `ggml-alloc` (정적 텐서 할당기)을 통해 추론 실행에 필요한 전체 메모리 버퍼 오프셋 계산 및 일괄 할당.

#### 적용 기술 및 컴퓨터 공학 이론
* **Static Memory Allocation & Buffer Reuse (정적 메모리 할당)**:
  * Computational Graph의 토폴로지 순서(Topological Order)를 분석하여 더 이상 필요하지 않은 중간 텐서의 버퍼 오프셋을 후속 텐서가 재사용(Buffer Lifetime Reuse)하도록 할당합니다.
  * 추론 실행 중 `malloc`/`free` 형태의 동적 힙 할당이 전혀 발생하지 않아 메모리 단편화(Fragmentation)와 실행 지연(Latency Spikes)을 완전히 제거합니다.

---

### 단계 3: 입력 처리 및 양자화 가중치 연산 준비 (Input Processing & Quantization)

#### 처리 순서
1. 문자열 프롬프트를 BPE / SentencePiece 토크나이저로 정수 토큰 ID 배열로 변환.
2. Embedding Table에서 해당 토큰의 임베딩 벡터 추출.
3. 가중치 양자화 상태 확인 및 동적 활성화 양자화(On-the-fly Quantization) 준비.

#### 적용 기술 및 수학적 이론

* **블록 양자화 (Block Quantization: Q4_0, Q4_1, Q8_0)**:
  * **역양자화 공식**: $x_i \approx s \cdot q_i + m$
  * **양자화 공식**: $q_i = \text{round}\left( \frac{x_i - m}{s} \right)$
    *(여기서 $s = \frac{x_{\max} - x_{\min}}{2^b - 1}$ 은 Scale, $m$ 은 Zero-point)*
  * 32개 원소(Block Size $k=32$) 단위로 FP16 Scale $s$를 두어, 이상치(Outlier)에 따른 정밀도 손실을 32개 국소 영역으로 제한합니다.

* **k-quants (Super-block Quantization: Q2_K ~ Q6_K)**:
  * 256개 원소를 포함하는 Super-block 구조 도입:
    $$x_i \approx S \cdot d_j \cdot (q_i - z_j)$$
    *(여기서 $S$는 Super-block의 FP16 Scale, $d_j$는 Sub-block의 4/6-bit Scale)*
  * 메타데이터 오버헤드를 가중치당 0.5bit 수준으로 극소화합니다.

* **IQ 양자화 (Importance Matrix & Lattice Vector Quantization: IQ1_S ~ IQ4_XS)**:
  * **Hessian Importance Matrix**: 손실 함수 $L$의 2차 테일러 전개 $\Delta L \approx \frac{1}{2} (W - \hat{W})^T H (W - \hat{W})$를 이용해 중요한 가중치를 보존합니다.
  * **$E_8$ Lattice Vector Quantization**: 8차원 공간 격자 코드북을 사용하여 1.5~2.5비트 수준으로 가중치를 압축합니다.

* **On-the-fly 활성화 양자화 (Q8_0 / Q8_K)**:
  * 가중치 $W$(Q4)와 연산할 입력 벡터 $x$(FP32)를 연산 직전에 Q8_0(8-bit 정수)으로 동적 양자화합니다.
  * 가중치를 FP32로 역양자화하지 않고, 정수 상태에서 직접 SIMD 점곱을 수행합니다.

---

### 단계 4: 트랜스포머 레이어 연산 (Transformer Layer Execution)

각 트랜스포머 블록에서는 다음 순서로 연산이 수행됩니다:

#### ① RMSNorm (Root Mean Square Normalization)
* **수학적 정의**:
  $$\text{RMSNorm}(x)_i = \frac{x_i}{\text{RMS}(x)} g_i, \quad \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{j=1}^d x_j^2 + \epsilon}$$
* **이론적 특징**: LayerNorm과 달리 평균 차감($x_i - \mu$) 절차가 필요 없어 1-pass 연산으로 완료되며 CPU 캐시 적중률이 상승합니다.

#### ② Q, K, V Projection 및 SIMD 가속 점곱 연산
* **수학 연산**: $Q = W_q \cdot x, \quad K = W_k \cdot x, \quad V = W_v \cdot x$
* **CPU SIMD 벡터화 기술**:
  * **x86-64**: AVX2(`__m256`), AVX-512/VNNI(`vpdpbusd` 정수 8비트 4쌍 1클록 점곱), AMX(2D 타일 행렬 연산).
  * **ARM**: Neon(`v128`), DotProd(`vdotq_s32` 8비트 정수 점곱), SVE/SVE2, i8MM.
  * **FMA (Fused Multiply-Add)**: $a \cdot b + c$를 단일 인스트럭션으로 처리.

#### ③ RoPE (Rotary Position Embedding) 위치 인코딩
* **수학적 정의**: 2D 벡터 평면 상에서의 회전 변환
  $$R_{\Theta, m}^{(2i, 2i+1)} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$
* **최적화**: $\sin, \cos$ 테이블 미리 계산 및 SIMD 가상 회전 연산으로 CPU 계산 병목 제거.

#### ④ KV Cache 저장 및 FlashAttention
* **Online Softmax 기반 CPU FlashAttention**:
  * **수학적 이론**: $N \times N$ 전체 Attention 행렬을 RAM에 생성하지 않고 블록 단위로 순회 연산:
    $$m_i = \max(m_{i-1}, x_i)$$
    $$\tilde{d}_i = \tilde{d}_{i-1} \cdot e^{m_{i-1} - m_i} + e^{x_i - m_i}$$
  * **효과**: RAM 사용량을 $O(N^2)$에서 $O(N)$으로 축소하여 L1/L2 캐시 메모리 내부에서 어텐션을 완결합니다.

#### ⑤ Feed-Forward Network (FFN / SwiGLU)
* **수학 연산**: $\text{FFN}(x) = W_{down} \cdot \left( \text{SiLU}(W_{gate} \cdot x) \odot (W_{up} \cdot x) \right)$
* **SiLU 활성화 함수 근사**:
  $$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$
  * CPU 지수 연산($e^{-x}$) 비용을 낮추기 위해 AVX2/Neon 레지스터 상에서 다항식 근사(Taylor/Padé Expansion) 적용.

---

### 단계 5: 토큰 생성 연산의 컴퓨터 공학적 특성 (Prefill vs Decode)

#### ① Prefill 단계 (GEMM: Matrix-Matrix Multiplication)
* **연산 특성**: 프롬프트 토큰 $N$개 ($N > 1$) 병렬 처리.
* **성능 병목**: **Compute-bound** (CPU의 초당 연산 처리 능력 FLOPS가 속도를 결정).
* **최적화**: Llamafile `sgemm.cpp` 및 Arm KleidiAI 마이크로 커널을 통한 L1/L2 캐시 타일링(Blocking) 및 루프 언롤링.

#### ② Decode 단계 (GEMV: Matrix-Vector Multiplication)
* **연산 특성**: 1개 토큰 단위 순차적 자동거듭생성(Autoregressive Generation).
* **성능 병목**: **Memory Bandwidth-bound** (CPU 연산 능력보다 DRAM 메모리 대역폭이 속도를 결정).
* **생성 속도 수학 모델**:
  $$\text{Generation Speed (tokens/sec)} \approx \frac{\text{System Memory Bandwidth (GB/s)}}{\text{Model Size (GB)}}$$
  * 예: 대역폭 50 GB/s 환경에서 10GB Q4 모델 사용 시 $\approx 5 \text{ tokens/sec}$, 양자화로 5GB 모델 사용 시 $\approx 10 \text{ tokens/sec}$로 속도가 2배 향상됩니다.

---

### 단계 6: 로짓 계산 및 샘플링 (Logits & Sampling)

#### 처리 순서
1. 마지막 레이어의 Output 벡터에 RMSNorm 적용.
2. `LM Head` 가중치 행렬 $W_{lm\_head}$와의 곱셈을 통해 보캐블러리 크기($V$)의 Logits 벡터 생성.
3. Temperature, Top-K, Top-P (Nucleus), Min-P, Repetition Penalty 필터링 적용.
4. Softmax 확률 분포 계산 후 최종 다음 토큰 ID 샘플링.

---

## 4. 핵심 기술 및 이론 요약표

| 구분 | 단계 | 핵심 기술 (Technology) | 적용 이론 및 수학적 원리 (Theory & Math) |
|---|---|---|---|
| **메모리** | 로딩 / 할당 | `mmap`, `ggml-alloc` | Zero-Copy, Virtual Memory Paging, Static Graph Buffer Lifetime Analysis |
| **스레딩** | 실행 관리 | Lock-free Threadpool, Core Affinity | Work-Stealing, NUMA Node Binding, P/E-core Scheduling |
| **양자화** | 가중치 압축 | Block, K-quants, IQ-quants | Affine Quantization, Hessian Importance Matrix, $E_8$ Lattice Vector Quantization |
| **연산** | 행렬 곱셈 | Dynamic Q8_0 Quantization | On-the-fly Activation Quantization, Integer SIMD Dot Product |
| **SIMD** | 하드웨어 가속 | AVX2, AVX-512/VNNI, ARM DotProd/SVE | Vectorization, Fused Multiply-Add (FMA), VNNI `vpdpbusd` Instruction |
| **커널** | GEMM/GEMV | Llamafile SGEMM, Arm KleidiAI | L1/L2 Cache Tiling, Loop Unrolling, Memory Alignment |
| **트랜스포머**| 레이어 연산 | RMSNorm, RoPE, FlashAttention | Single-pass RMS Normalization, Complex Matrix Rotation, Online Stable Softmax |
| **성능 모델**| Decode 병목 | Quantization Throughput Scaling | Memory Bandwidth Bound Model ($\text{Speed} \propto \frac{\text{Bandwidth}}{\text{Model Size}}$) |

---
*작성일: 2025년*  
*참조: llama.cpp / ggml-cpu 아키텍처 및 소스 코드 연산 구조*
