# rag-adk-pinecone

Google ADK와 Pinecone 벡터 데이터베이스를 활용한 RAG(Retrieval-Augmented Generation) 구현

## 개요

이 저장소는 Google의 Agent Development Kit(ADK)와 효율적인 벡터 검색을 위한 Pinecone을 사용하여 RAG 시스템을 구축하는 방법을 보여줍니다. 이 구현은 외부 지식 소스를 참조할 수 있는 컨텍스트 인식 AI 애플리케이션을 개발하기 위한 기반을 제공합니다.

주요 기능:
- Google ADK와 Pinecone 벡터 데이터베이스 통합
- 문서 인덱싱 및 검색 기능
- RAG 기능을 활용하는 예제 에이전트
- 데이터 전처리를 위한 PDF 문서 업로더

## 프로젝트 구조

```
rag-adk-pinecone/
├── rag/
│   ├── __init__.py
│   ├── agent.py            # 루트 에이전트 정의
│   └── tools/
│       └── tools.py        # PineconeIndexRetrieval 도구 구현
├── .env.example            # 환경 변수 템플릿
├── .gitignore
├── LICENSE
├── pdf_uploader.py         # PDF를 벡터 데이터베이스에 업로드하는 유틸리티
├── README.md
└── requirements.txt
```

## 사전 요구사항

- Python 3.9+
- Pinecone 계정 및 API 키
- OpenAI API 키
- Google AI API 키가 설정된 Google ADK
- 필요한 Python 패키지 (requirements.txt에 나열됨)

## 설치

1. 저장소 클론:
   ```
   git clone https://github.com/jeyong-shin/rag-adk-pinecone.git
   cd rag-adk-pinecone
   ```

2. 환경 템플릿을 복사하고 자격 증명 입력:
   ```
   cp .env.example .env
   ```
   
   `.env` 파일을 편집하여 다음 항목을 포함:
   - API 키
   - Pinecone 환경
   - Pinecone 인덱스 이름
   - 기타 필요한 구성

3. 가상 환경 생성 및 활성화:
   ```
   python -m venv venv
   ```
   
   Linux/macOS:
   ```
   source venv/bin/activate
   ```
   
   Windows:
   ```
   .\venv\Scripts\Activate.bat
   ```

4. 의존성 설치:
   ```
   pip install -r requirements.txt
   ```

## 사용법

### 문서 업로드

에이전트를 실행하기 전에 Pinecone 인덱스에 문서를 업로드할 수 있습니다:

```
python pdf_uploader.py 문서/경로/파일명.pdf 네임스페이스_이름
```

이 유틸리티는 다음 작업을 수행합니다:
1. PDF에서 텍스트 추출
2. 텍스트를 관리 가능한 청크로 분할
3. 각 청크에 대한 임베딩 생성
4. 임베딩을 Pinecone 인덱스에 업로드

### 애플리케이션 실행

ADK 웹 인터페이스 시작:

```
adk web
```

이렇게 하면 RAG 에이전트와 상호작용할 수 있는 로컬 웹 서버가 시작됩니다.

## 작동 방식

1. **문서 인덱싱**: 문서가 처리되어 청크로 분할되고 Pinecone의 벡터 데이터베이스에 저장됩니다.

2. **쿼리 처리**: 사용자가 쿼리를 제출하면:
   - `PineconeIndexRetrieval` 도구가 쿼리를 임베딩으로 변환
   - Pinecone 인덱스에서 관련 문서 청크를 검색
   - 의미적 유사성을 기준으로 가장 관련성 높은 청크를 검색

3. **응답 생성**: `rag/agent.py`의 루트 에이전트는 검색된 컨텍스트와 원래 쿼리를 함께 사용하여 정보에 기반한 응답을 생성합니다.

## 커스터마이징

### RAG 에이전트 수정

주요 에이전트 로직은 `rag/agent.py`에 정의되어 있습니다. 이 파일을 수정하여:
- 프롬프트 템플릿 조정
- 검색 전략 변경
- 추가 비즈니스 로직 구현

### 검색 도구 확장

`rag/tools/tools.py`의 `PineconeIndexRetrieval` 도구는 다음과 같이 확장할 수 있습니다:
- 다양한 임베딩 모델 지원
- 고급 필터링 구현
- 메타데이터 기반 검색 기능 추가

## 문제 해결

### 일반적인 문제

- **인증 오류**: `.env` 파일의 API 키가 올바른지 확인
- **인덱스를 찾을 수 없음**: Pinecone 인덱스 이름이 `.env` 파일의 이름과 일치하는지 확인
- **임베딩 오류**: 호환되는 임베딩 모델을 사용하고 있는지 확인

## 라이선스

자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 감사의 말

- Google Agent Development Kit (ADK)
- Pinecone Vector Database
- LangChain