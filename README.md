# Live KR→EN 발표 자막 오버레이 (Google Cloud 기반)

PPT 발표 중 **마이크 음성(한국어)** 을 실시간으로 인식해서 **영문 자막(오버레이)** 로 띄우는 프로그램입니다.

핵심 설계는 “최소 지연 + 높은 자연스러움”:
1) **대본(한글) ↔ 사전 번역(영문)** 을 미리 넣어두고  
2) 실시간 STT 결과가 대본과 **의미/유사도 매칭** 되면 → **미리 번역된 자막을 즉시 출력**  
3) 대본에 없는 발화는 → **Cloud Translation(용어집 Glossary 적용 가능)** 으로 번역해 출력

> Speech-to-Text **V2 StreamingRecognize는 gRPC 양방향 스트리밍에서만 지원**됩니다. :contentReference[oaicite:0]{index=0}  
> Translation은 **Glossary로 용어 번역을 고정**할 수 있습니다. :contentReference[oaicite:1]{index=1}  
> Codex CLI는 로컬 디렉토리에서 코드를 읽고/수정하고/실행까지 돕는 CLI 에이전트입니다. :contentReference[oaicite:2]{index=2}

---

## 1. 목표 기능 (MVP → 확장)

### MVP
- [ ] 마이크 입력 → **Google Speech-to-Text v2 스트리밍** → 실시간 한국어 자막(콘솔)
- [ ] PyQt6 **항상 위(Always-on-top) 투명 오버레이**로 자막 표시

### 발표 친화 기능
- [ ] (중요) **대본 의미 매칭**: 대본 구간은 “미리 번역된 영문 자막”을 즉시 출력
- [ ] 대본 밖 발화는 **Cloud Translation v3**로 번역하여 출력 (선택: Glossary 적용)
- [ ] STT **interim(중간) 결과**를 빠르게 표시하고, **final(확정) 결과**로 문장 다듬기
- [ ] STT **voice activity events**로 말 시작/끝을 감지하여 문장 확정 타이밍 개선 :contentReference[oaicite:3]{index=3}
- [ ] 용어 힌트(phrase sets/adaptation)로 천문/ML 키워드 인식률 개선 :contentReference[oaicite:4]{index=4}

---

## 2. 전체 아키텍처

Mic Audio
↓ (chunk, 20~50ms frames)
Audio Capture (sounddevice / pyaudio)
↓
Google Speech-to-Text v2 (gRPC StreamingRecognize)
↓ ↘ (voice activity events)
KR transcript (interim/final)
↓
Script Matcher (windowed search + semantic/fuzzy score)
├─ match ✅ → use pretranslated EN line (instant)
└─ no match ❌ → Cloud Translation v3 → EN subtitle (optional Glossary)
↓
Overlay UI (PyQt6 transparent always-on-top)

---

## 3. 기술 스택

- Python 3.10+
- 오디오 입력: `sounddevice` (권장) 또는 `pyaudio`
- UI 오버레이: `PyQt6`
- STT: `google-cloud-speech` (Speech-to-Text **v2**, gRPC 스트리밍) :contentReference[oaicite:7]{index=7}
- 번역: `google-cloud-translate` (Translation v3, Glossary 지원) :contentReference[oaicite:8]{index=8}
- 대본 매칭:
  - 빠른 문자열 유사도: `rapidfuzz`
  - 의미 임베딩(선택): `sentence-transformers` (멀티링구얼 임베딩)

---

## 4. 레포 구조(권장)

.
├─ README.md
├─ requirements.txt
├─ .env.example
├─ data/
│ ├─ script.csv # 대본 (ko/en + keywords)
│ └─ glossary.csv # (선택) 번역 용어집 원본
├─ app/
│ ├─ main.py # 엔트리포인트
│ ├─ audio_io.py # 마이크 캡처/버퍼
│ ├─ stt_google_v2.py # gRPC 스트리밍 STT
│ ├─ matcher.py # 대본 매칭 로직
│ ├─ translate_google_v3.py # 번역(+glossary 옵션)
│ ├─ overlay_qt.py # 오버레이 UI
│ └─ config.py # env/config 로딩
└─ scripts/
├─ create_recognizer.py # (선택) recognizer 생성 유틸
└─ create_glossary.py # (선택) glossary 생성 유틸

---

## 5. Google Cloud 준비

### 5.1 프로젝트/과금
1) Google Cloud 프로젝트 생성
2) Billing 연결(무료 포기 조건)

### 5.2 API 활성화
- Speech-to-Text API
- Cloud Translation API

(예시)
```bash
gcloud services enable speech.googleapis.com translate.googleapis.com
5.3 서비스 계정 & 키(JSON)

서비스 계정 생성

권한 부여(예시)

Speech-to-Text 사용 권한

Translation 사용 권한

키(JSON) 생성 후 다운로드

로컬에서:
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

5.4 Speech-to-Text v2 Recognizer 생성

Speech-to-Text v2는 “Recognizer” 리소스를 사용합니다.
REST로 생성 가능(형식/엔드포인트는 create 문서 참고).

실시간 스트리밍은 gRPC StreamingRecognize로 호출합니다.

5.5 (선택) Translation Glossary 준비

Glossary는 용어 번역을 고정합니다.
코드 샘플/클라이언트는 Google 문서 및 Python 레퍼런스를 참고합니다.

6. 로컬 설치/실행
6.1 Python 환경
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
6.2 환경 변수(.env)

.env.example를 복사해 .env로 만들고 채우세요.

필수(예시):

GOOGLE_APPLICATION_CREDENTIALS

GCP_PROJECT_ID

GCP_LOCATION (예: global)

GCP_RECOGNIZER
예: projects/{PROJECT_ID}/locations/{LOCATION}/recognizers/{RECOGNIZER_ID}

선택:

GCP_TRANSLATE_LOCATION (보통 global)

GCP_GLOSSARY_ID

SCRIPT_PATH (대본 CSV 경로)

6.3 실행
python -m app.main

7. 대본 데이터 포맷

data/script.csv (UTF-8)

id,ko,en,keywords
0001,"오늘 발표의 목표는 ...","Today, the goal of this talk is ...","목표;goal;overview"
0002,"데이터셋은 ...","Our dataset is ...","dataset;sample;survey"


권장 규칙:

keywords는 세미콜론(;)으로 구분

발표 흐름을 살리려면 id 순서가 발표 순서가 되도록 정렬

8. 대본 매칭(권장 알고리즘)

발표는 대개 순서대로 진행되므로:

현재 포인터 i를 두고 i~i+K(예: 10~30줄)만 후보로 평가

점수(예시):

의미 유사도(임베딩 cosine)

문자열 유사도(rapidfuzz)

키워드 히트(전문용어)

순서 prior(가까운 줄 가산점)

임계값 이상이면:

미리 번역된 en을 바로 출력(최저 지연/최고 자연스러움)


10. 품질/지연 튜닝 체크리스트

STT:

interim results 사용 (UI는 interim을 즉시 보여주고 final로 확정)

voice activity events로 문장 경계 안정화

Phrase/adaptation으로 용어 인식 강화

매칭:

후보를 i~i+K로 제한(오탐↓, 속도↑)

“최근 2~4초 누적 텍스트”로 매칭(부분 인식 흔들림 완화)

번역:

Glossary로 핵심 용어 고정

11. 트러블슈팅

STT v2 스트리밍이 안 됨:

v2 StreamingRecognize는 gRPC만 지원(REST로는 불가)

GOOGLE_APPLICATION_CREDENTIALS 경로/권한 확인

Recognizer 리소스 경로 확인(프로젝트/리전/ID)

용어 번역이 Glossary로 안 고정됨:

번역 응답 필드에서 glossary 번역 결과를 올바르게 사용해야 함(문서/샘플 확인)

12. 다음 단계(원하면 바로 구현)

 OBS 브라우저 소스용 WebSocket/HTML 오버레이 모드 추가

 자동 줄바꿈/자막 길이 제한(예: 42 chars)

 발표 모드: “대본 구간만 출력(완전 고품질)” 옵션

 다국어(ko→en 외) 확장

참고 문서(핵심)

OpenAI Codex CLI Quickstart / CLI Reference

Speech-to-Text v2 gRPC StreamingRecognize

Speech-to-Text Recognizers 개념

Voice activity events 샘플

Translation Glossary