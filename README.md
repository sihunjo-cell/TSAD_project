# TSAD_project

정상 데이터가 거의 없는 제조 현장에서 **어떤 TSAD 모델을 언제 도입하고, 유지하거나 교체할지** 결정하는 운영 계획을 설계한다.

데이터가 쌓이는 동안의 성능과 학습·추론 비용을 비교하며, 오탐·미탐·데이터 대기·자원 비용까지 반영하는 것을 목표로 한다.

## 연구 흐름

```text
데이터 EDA·manifest 정리
→ 모델 구현·실행 조건 확인
→ Dev18 튜닝 실행·점수 및 비용 기록
→ VUS-PR 채점·family-LOFO 검증·설정 선택
→ SQLite DB 구성·Streamlit 연결
→ 데이터 축적에 따른 모델 선택·운영 경로 설계
→ GHL25 주실험·HAI 외부 확인
```

본실험 대상은 **GHL과 HAI**다. `TSB-AD-M-Tuning.csv`에서 GHL을 제외한 18개 파일은 모델과 recipe를 정하는 사전 튜닝 패널이다. `dev18`은 이 패널의 내부 식별자이며 별도 본실험 데이터셋을 뜻하지 않는다.

모델 실행과 결과 생성은 Lightning AI 서버에서 진행한다. Tier 2는 PaAno와 공식 `d-ailin/GDN` 적응 구현을 사용하며, ALoRa는 후보에서 제외한다.

## 운영 계획 파이프라인

Streamlit은 정상 데이터와 운영 조건을 받아 DB의 등록 후보를 정리한다. 이를 출발점으로 하는 추천·최적화 절차는 아래와 같이 설계한다.

```text
누적 정상 prefix + 현재 모델 정보 + 운영 조건 입력
↓
SQLite에서 모델·설정·head 후보 조회
↓
고정된 센서 구성과 모델의 필수 조건으로 1차 축소
↓
현재 prefix의 패턴 특징과 비슷한 과거 Dev18 prefix 검색
↓
해당 과거 파일의 이후 prefix 기록까지 연결
↓
미래 단계별 확보 행 수·추론량 계산
→ 단계별 실행 조건 확인
→ 후보별 현재·미래 성능과 학습·추론 비용 추정
↓
단계별 신규 도입·재학습 후보 k개 선정
기존 checkpoint의 유지 선택은 별도로 보존
↓
DP로 유지·재학습·교체 경로 선택
↓
현재 행동 적용·운영 상태 기록
↓
새 누적 prefix가 들어오면 전체 계획 후보 풀에서 재평가
→ 남은 운영 기간의 경로 갱신
```

### 입력과 1차 축소

- 갱신할 때는 **기존 데이터를 포함한 누적 정상 prefix 전체**를 입력한다. 이전 입력에 이어 붙이지 않고 새 입력으로 교체한다.
- 운영 기간 동안 센서 구성이 같다고 가정한다. 현재 모델·설정·checkpoint·마지막 학습 시점과 장비·성능 하한·운영 기간·수집속도·추론량을 함께 받는다.
- 1차 축소는 센서 수와 고정된 필수 입력 조건만 확인한다. 현재 학습 길이·유효 창 수·상수 여부·분산·상관·결측으로 미래 후보까지 영구 제외하지 않는다.
- 구조 조건을 확인할 recipe 설정이 없으면 계획 후보에서 제외한다. 과거 실행 이력이나 성능 점수는 1차 축소 기준으로 쓰지 않는다.

> 계획 후보에 포함됐다는 것은 즉시 실행 가능하다는 뜻이 아니다. 실제 학습량과 장비·추론량에 따른 조건은 단계별로 확인한다.

### 유사도와 성능·비용 추정

유사도는 변동성·자기상관·채널 간 상관·분포 변화·spectral entropy 등 패턴 특징으로 계산한다. 특징 표준화와 Euclidean 거리, 거리 가중 kNN을 첫 비교안으로 둔다.

**행 수는 유사도 입력과 분리**해 실행 조건·성능 변화·비용 추정에 사용한다. 과거 파일과 행 수가 다르다는 이유만으로 제외하지 않으며, 서로 다른 파일의 같은 q를 같은 학습량으로 취급하지 않는다.

미래 계획은 *데이터 발생 특성이 크게 변하지 않는 기본 시나리오*를 사용한다. DB 관측 범위를 넘는 성능과 기존 checkpoint의 유지 성능은 마지막 기준 수준을 유지한다고 가정한다. 비용은 모델별 학습량·구간별 추론량·장비 조건에 맞춰 별도로 추정한다.

단계별 후보 수는 `k=10`처럼 정하되(예시), 후보 구성은 단계마다 달라진다. 유지 선택은 이 축소와 별도로 남긴다.

### 경로 최적화

DP의 상태는 `단계·모델·설정·head·마지막 학습 단계`로 구성한다. 최초 도입, 유지, 재학습, 교체를 비교해 **예상 성능 하한을 만족하면서 전체 학습·추론비가 가장 작은 경로**를 고른다.

유지에는 해당 구간의 추론비를, 도입·재학습·교체에는 필요한 학습비와 추론비를 반영한다. 별도 전환비는 초기 목적함수에서 제외한다.

결과는 선택한 후보와 미래 가정 안에서의 조건부 계획이며, 새 데이터가 들어오면 이전 top-k에서 빠졌던 후보까지 다시 검토한다.

## Streamlit 실행

서버에서 프로젝트 루트를 기준으로 실행한다.

```bash
python -m pip install -r requirements.txt
cd streamlit_website
python -m streamlit run app.py
```

정상 구간 CSV와 센서 열을 선택한 뒤 현재 모델과 운영 조건을 입력한다. 화면은 현재 데이터 특징과 구조 조건을 통과한 계획 후보를 표시한다.

`streamlit_website/db_connection/`은 아래 DB를 **읽기 전용**으로 조회한다.

```text
experiments/tuning/results/recommendation.sqlite3
```

입력 데이터, 계산한 특징, 현재 모델 상태, 운영 조건, 통과 후보와 원래 recipe 설정은 `st.session_state["ml_input"]`에 묶어 다음 단계의 전달 형식으로 둔다.

입력은 현재 세션 메모리에서 관리하며, 새 파일을 올리면 이전 전달값을 지우고 제출할 때 교체한다.

> [!IMPORTANT]
> 실행 환경에 `recommendation.sqlite3`가 있어야 한다. DB와 튜닝 결과는 실행 산출물이므로 Git 추적에서 제외한다.

## 폴더 구성

```text
configs/ + src/
      ↓
tests/에서 실험 실행
      ↓
experiments/에 점수·실행 증거·결과·DB 저장
      ↓
streamlit_website/db_connection/에서 DB 조회
      ↓
Streamlit 입력과 계획 후보 연결
```

| 경로 | 역할 |
| --- | --- |
| `streamlit_website/app.py` | Streamlit 진입점과 화면 구성 |
| `streamlit_website/db_connection/` | CSV 입력, DB 조회, 고정 제약 필터, 다음 단계 전달값 구성 |
| `streamlit_website/.streamlit/` | Streamlit 설정 |
| `configs/` | 데이터·모델·채점·운영 시나리오 설정 |
| `src/models/` | Tier 1·2·3 모델 |
| `src/data_split/` | 데이터 로더와 prefix 분할 |
| `src/common/` | 공통 실행, 특징 계산, 실행 조건 검사, 점수·증거 저장 |
| `src/채점기/` | VUS-PR 채점 |
| `tests/tuning/` | Dev18 튜닝·예산·채점·선택·DB 기록 |
| `tests/ghl_main/` | GHL 및 공통 모델 실행·출력 검사 |
| `tests/hai_extension/` | HAI 외부 확인 실험 영역 |
| `tests/checks/` | 데이터·환경·참고 구현 점검 |
| `tests/unit/` | 재사용하는 단위 검증 |
| `experiments/` | 실험 산출물 |

실험별 산출물 루트는 다음과 같다.

| 실험 | 산출물 경로 |
| --- | --- |
| Dev18 튜닝 | `experiments/tuning/` |
| GHL 주실험 | `experiments/01_ghl_main/` |
| HAI 외부 확인 | `experiments/02_hai_extension/` |
| 사전 점검 | `experiments/checks/` |

튜닝 결과와 DB는 `experiments/tuning/results/`, 예산과 기준 파일은 `experiments/tuning/snapshots/`에 바로 저장한다. 점수와 실행 로그의 경로는 각각 `scores/`, `logs/`다. 결과 안의 `figures/`, `tables/`, `metadata/` 등 내용별 하위 폴더는 유지한다.

`docs/`에는 로컬 `superpowers/` 지침을 두며 Git으로 추적하지 않는다. 원본 데이터 경로인 `TSB-AD-M/`, `TSB-AD-M-other-datasets/`, `HAI-23.05/`도 Git 추적 대상에서 제외한다.

## 모델 실행

Streamlit 의존성은 루트의 `requirements.txt`, 모델 실행 의존성은 `src/models/requirements.txt`로 관리한다.

Lightning AI 서버에서 프로젝트 루트를 기준으로 튜닝 준비를 실행한다.

```bash
python -m tests.tuning.run_ratio_tuning --prepare
```

원격 검증과 예산 봉인을 마친 뒤 `--prepare`를 빼면 Dev18 패널을 재개형으로 실행하고, 채점·family-LOFO 선택·final membership과 표·그림을 생성한다. GHL25와 HAI는 선택표 봉인과 데이터 인수 검사를 거쳐 평가한다.

| 실행 옵션 | 용도 |
| --- | --- |
| `--prepare` | 실행 계획과 저장 계약 준비 |
| `--finish-only --remote-cpu` | 원격 CPU에서 채점·선택·결과 정리 |
| `--selection-only --remote-cpu` | 원격 CPU에서 기존 채점 결과로 선택 재실행 |

> [!NOTE]
> 고비용 실행은 CUDA 또는 명시한 원격 환경에서 수행한다. 로컬 노트북에서는 모델 실행·학습·전체 테스트를 돌리지 않으며, CUDA를 사용할 수 없으면 CPU로 자동 전환하지 않는다.

기존 결과와 봉인 정보는 보존한다. 코드나 경로 변경 후의 실행 검증과 재봉인은 Lightning AI에서 진행한다.