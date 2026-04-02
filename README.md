# llama.cpp (Forked for Android/Termux Optimization)

이 프로젝트는 [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)를 포크한 저장소입니다.  
주요 목적은 **Android(Termux)** 환경에서 LLM(Large Language Model)을 효율적으로 구동하기 위한 빌드 최적화 및 성능 테스트입니다.

---

## 📌 출처 및 원본 프로젝트 (Original Source)

모든 핵심 소스 코드와 로직의 저작권은 원본 작성자에게 있습니다. 원본 프로젝트에 대한 상세한 정보와 최신 업데이트는 아래 링크를 참조하세요.

* **Original Repository:** [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
* **Original README:** [View Original Documentation](https://github.com/ggerganov/llama.cpp/blob/master/README.md)
* **License:** MIT License

---

## 📱 현재 타겟 환경 (Target Environment)

본 포크 저장소는 특히 아래의 하드웨어 사양에 최적화된 빌드 환경을 구성하는 데 집중하고 있습니다.

* **Device:** Samsung Galaxy Fold 3
* **Processor:** Qualcomm Snapdragon 888
* **GPU:** Adreno 660 (Vulkan / OpenCL 지원 테스트)
* **Memory:** 12GB RAM
* **OS:** Android (Termux 환경)

---

## 🛠️ 주요 수정 및 테스트 사항 (Work in Progress)

원본 프로젝트를 바탕으로 다음 항목들을 중점적으로 다룹니다.

1.  **Build Optimization:** Termux 내에서 CPU 전용, Vulkan 및 OpenCL 백엔드 빌드 시 발생하는 오류 해결.
2.  **Performance Tuning:** Snapdragon 888의 Adreno GPU를 활용한 추론 속도 최적화.
3.  **Model Compatibility:** GGUF 형식의 다양한 모델(Gemma, Qwen, Llama 등) 구동 테스트.

---

## 🚀 빠른 시작 (Quick Start)

Termux 환경에서의 기본적인 빌드 절차는 다음과 같습니다 (원본 가이드 준수).

### CPU 전용 빌드
```bash
cmake -B build
cmake --build build --config Release
