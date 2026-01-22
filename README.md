# 🤖 SBERT 기반 지능형 AI 챗봇 & 의도 분류기
> **Pre-trained Korean Embedding Model을 활용한 FAQ 챗봇 및 문장 분석 시스템**

![Generic badge](https://img.shields.io/badge/Language-Python_3.9+-3776AB.svg)
![Generic badge](https://img.shields.io/badge/Library-Streamlit-FF4B4B.svg)
![Generic badge](https://img.shields.io/badge/AI_Model-KoSBERT-orange.svg)

## 📖 프로젝트 소개 (Overview)
이 프로젝트는 **한국어 문장 임베딩(Sentence Embedding) 모델**을 활용하여, 사용자의 질문 의도를 파악하고 적절한 답변을 제공하는 **AI 챗봇 애플리케이션**입니다.

대규모 LLM(ChatGPT 등)을 사용하지 않고, **SBERT(Sentence-BERT)** 모델과 **코사인 유사도(Cosine Similarity)** 알고리즘만으로 **빠르고 가벼운 온디바이스(On-Device)급 챗봇**을 구현했습니다. `Streamlit`을 통해 웹 브라우저에서 즉시 테스트가 가능합니다.

---

## ✨ 핵심 기능 (Key Features)

### 1. 💬 AI FAQ 챗봇 (Similarity Search)
* **기능:** 사용자가 질문을 입력하면, 사전에 정의된 `company_docs.csv` 데이터베이스에서 가장 의미가 유사한 질문을 찾아 답변을 출력합니다.
* **알고리즘:**
  1. 사전 학습된 `jhgan/ko-sroberta-multitask` 모델로 질문을 벡터화(Embedding)합니다.
  2. DB 내의 질문 벡터들과 **코사인 유사도(Cosine Similarity)**를 계산합니다.
  3. 유사도가 **0.5 이상**인 경우 가장 적합한 답변을 반환하고, 미만일 경우 "이해하지 못했습니다"라며 답변을 보류하는 **신뢰도 필터링(Confidence Filtering)**이 적용되어 있습니다.

### 2. 🧠 제로샷 의도 분류 (Zero-Shot Intent Classification)
* **기능:** `sentences.csv`에 있는 문장 패턴을 학습하여, 사용자가 입력한 문장이 어떤 카테고리(예: 배송 문의, 환불 요청 등)에 속하는지 분류합니다.
* **특징:** 별도의 복잡한 학습 과정(Fine-tuning) 없이, 임베딩 거리 계산만으로 문장의 의도를 파악하는 **Semantic Search** 기법을 사용했습니다.

### 3. 🖥️ 대화형 웹 인터페이스 (Streamlit UI)
* **기능:** 파이썬 코드만으로 간편하게 웹 애플리케이션을 구축할 수 있는 `Streamlit`을 사용하여, 실제 채팅 앱과 유사한 UI를 제공합니다.
* **UX:** 사용자와 AI의 대화 내역(History)이 화면에 누적되어 자연스러운 대화 흐름을 경험할 수 있습니다.

---

## 🛠️ 기술 스택 (Tech Stack)

| 구분 | 기술 (Technology) | 설명 |
| :-- | :-- | :-- |
| **Language** | Python 3.9+ | 핵심 로직 구현 |
| **AI Model** | `jhgan/ko-sroberta-multitask` | 한국어 문장 임베딩 생성 (Hugging Face) |
| **Library** | `sentence-transformers` | SBERT 모델 구동 및 유사도 계산 |
| **Frontend** | `Streamlit` | 웹 대시보드 및 채팅 UI 구현 |
| **Data** | `Pandas` | CSV 데이터(FAQ, 문장셋) 처리 |

---

## 📂 파일 구조 및 데이터 설명

```bash
📦 AI-Chatbot-Project
 ┣ 📜 web_learner_bot.py   # 메인 실행 파일 (Streamlit App)
 ┣ 📜 company_docs.csv     # 챗봇 지식 베이스 (FAQ 데이터)
 ┗ 📜 sentences.csv        # 의도 분류용 학습 문장 데이터
