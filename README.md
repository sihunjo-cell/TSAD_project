# TSAD_project

제조 시계열 이상 탐지 모델을 세 계층으로 비교하는 프로젝트다. 모델 코드, 실험 코드,
실험 결과를 서로 다른 폴더에 둔다.

## 폴더 관계도

```text
docs/ ───── 계획·역할 ───┐
configs/ ── 실행 설정 ───┼─> tests/ ── 실행 ─> experiments/
src/ ───── 모델·공통 코드 ┘   실험 코드          결과
```

| 폴더 | 의미 | 연결되는 위치 |
| --- | --- | --- |
| [`configs/`](configs/) | 데이터 전처리와 모델·채점 설정 | `src/`, `tests/`가 읽음 |
| [`src/`](src/) | 모델 본체와 여러 실험이 함께 쓰는 코드 | `tests/`에서 호출 |
| [`tests/ghl_main/`](tests/ghl_main/) | GHL 본 실험 실행과 완료 검사 | [`experiments/01_ghl_main/`](experiments/01_ghl_main/) |
| [`tests/hai_extension/`](tests/hai_extension/) | HAI 확장 실험 실행과 완료 검사 | [`experiments/02_hai_extension/`](experiments/02_hai_extension/) |
| [`tests/checks/`](tests/checks/) | 다시 실행할 데이터·참고 구현 점검 | [`experiments/checks/`](experiments/checks/) |
| [`tests/unit/`](tests/unit/) | 모델과 공통 코드의 단위 검증 | 별도 결과를 보관하지 않음 |
| [`experiments/`](experiments/) | 점수·로그·스냅샷·표·그림 | 실행 코드를 두지 않음 |
| [`docs/`](docs/) | 계획, 현재 결정, 담당자별 작업 계약 | 프로젝트 진행 기준 |

`src/models/`는 계층별 모델, `src/data_split/`은 데이터 분할과 로더,
`src/common/`은 점수 저장·정규화·실행 검사를 맡는다.

## 담당자 문서

| 담당자 | 역할 | 바로가기 |
| --- | --- | --- |
| 저 | 계층 1·2·3 모델과 점수 생산, 전체 설계 관리 | [`docs/lead/`](docs/lead/) |
| 강혁 | GHL·HAI 데이터 검수와 Manifest | [`docs/role_A/`](docs/role_A/) |
| 지우 | VUS-PR 채점기와 ℓ_max | [`docs/role_B/`](docs/role_B/) |
| 주혜 | 난이도 분할, 통계, 결과 집필 | [`docs/role_C/`](docs/role_C/) |

처음에는 [연구 계획](docs/lead/plan_v4.md),
[사전 점검 결과](docs/lead/process_0_preverify.md),
[다음 세션 시작점](docs/lead/next_session.md) 순서로 읽는다.

## 실행 원칙

전체 단위 검증은 `python -m unittest discover -s tests`로 실행한다. 실데이터 학습은
[사전 점검 결과](docs/lead/process_0_preverify.md)의 실행 전 조건을 모두 충족한 뒤 시작한다.
