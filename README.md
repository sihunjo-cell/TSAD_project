# TSAD_project

심화세션 5조 시훈, 강혁, 지우, 주혜 프로젝트다. 정상 데이터가 거의 없는 제조 현장에서 어떤
TSAD 계층과 모델을 언제 도입할지 판단하고, 마지막에는 성능·오탐·미탐·데이터 대기·학습·추론·
자원 비용을 함께 넣은 목적함수로 운영안을 고른다.

본실험 데이터는 GHL과 HAI다. `TSB-AD-M-Tuning.csv`의 비-GHL 18개는 GHL을 보기 전에 모델과
recipe를 정하는 사전 튜닝 패널이다. 기존 파일명과 경로의 `dev18`은 이 패널의 내부 식별자이며
별도 본실험 데이터셋이 아니다.

## 연구 흐름

```text
강혁 EDA·manifest → 모델 담당자 정적 feasibility·모델 함수·점수
→ 지우 VUS-PR·튜닝 선택 → GHL25 주실험 → HAI 외부 확인
→ 주혜 통계·교차점 → 비용 목적함수와 최종 도입안
```

GDN은 공식 `d-ailin/GDN` 적응 구현 하나만 Tier 2 후보로 쓴다. 과거 HAI 전용 GDN 경로와
그래프 보조 분석은 현행 연구 범위가 아니다.

## 폴더 관계

```text
docs/ ───── 계획·역할 ───┐
configs/ ── 실행 설정 ───┼─> tests/ ── 실행 ─> experiments/
src/ ───── 모델·공통 코드 ┘   실험 코드          결과
```

| 폴더 | 의미 | 연결되는 위치 |
| --- | --- | --- |
| [`configs/`](configs/) | 데이터 전처리와 모델·채점 설정 | `src/`, `tests/`가 읽음 |
| [`src/`](src/) | 모델 본체와 여러 실험이 함께 쓰는 코드 | `tests/`에서 호출 |
| [`tests/ghl_main/`](tests/ghl_main/) | TSB 튜닝·GHL 본실험 실행과 완료 검사 | [`experiments/01_ghl_main/`](experiments/01_ghl_main/) |
| [`tests/hai_extension/`](tests/hai_extension/) | HAI 외부 확인 실행과 완료 검사 | [`experiments/02_hai_extension/`](experiments/02_hai_extension/) |
| [`tests/checks/`](tests/checks/) | 다시 실행할 데이터·참고 구현 점검 | [`experiments/checks/`](experiments/checks/) |
| [`tests/unit/`](tests/unit/) | 모델과 공통 코드의 단위 검증 | 별도 결과를 보관하지 않음 |
| [`experiments/`](experiments/) | 점수·로그·snapshot·표·그림 | 실행 코드를 두지 않음 |
| [`docs/`](docs/) | 계획, 현재 결정과 담당자별 인수 계약 | 프로젝트 진행 기준 |

`src/models/`는 계층별 모델, `src/data_split/`은 데이터 분할과 loader,
`src/common/`은 점수 저장·정규화·실행 검사를 맡는다.

## 역할

| 담당자 | 책임 | 문서 |
| --- | --- | --- |
| 시훈 | Tier 1·2·3 모델과 본실험 함수, 점수·실행 증거, 전체 설계 | [`docs/lead/`](docs/lead/) |
| 강혁 | GHL·HAI·TSB 튜닝 데이터 EDA와 manifest | [`docs/role_A/`](docs/role_A/) |
| 지우 | VUS-PR, `ℓ_max`, 튜닝 채점과 고정 선택표 | [`docs/role_B/`](docs/role_B/) |
| 주혜 | 난이도, 통계, 교차점과 비용 최적화용 결과 원표 | [`docs/role_C/`](docs/role_C/) |

처음에는 [계획서 v5](docs/lead/plan_v5.md),
[사전 점검](docs/lead/process_0_preverify.md),
[다음 세션 시작점](docs/lead/next_session.md) 순서로 읽는다.

## 현재 게이트

0·1단계를 마쳤고 TSB 비-GHL 18개 튜닝을 실행하기 직전이다. 공통 환경은
`C:\Users\simon\anaconda3\envs\tsad_models_311`이며 설치 선언은
[`src/models/requirements.txt`](src/models/requirements.txt) 하나로 관리한다.

준비 상태만 확인하려면 아래처럼 실행한다.

```powershell
C:\Users\simon\anaconda3\envs\tsad_models_311\python.exe tests\ghl_main\run_dev18_tuning.py --prepare
```

같은 파일을 옵션 없이 실행하면 18개 exact panel을 재개형으로 끝까지 돌리고, 채점·family-LOFO
선택·final membership과 모델별/Tier별 CSV·PNG를
`experiments/01_ghl_main/results/dev18_tuning/`에 저장한다. GHL25와 HAI는 이 선택표가 봉인된 뒤
각 데이터 인수 게이트를 거쳐 실행한다.

고비용 실행의 기본 device는 CUDA다. CUDA가 활성화되지 않은 로컬 노트북에서는 CPU로
되돌리지 않고 중단한다. 원격 CPU에서만 `--remote-execution --device cpu`를 명시해 실행한다.
