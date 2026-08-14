# TSAD_project

to-bigs 심화세션 시훈·강혁·지우·주혜 조의 미니 프로젝트 저장소다. 신규 설비 라인처럼 학습 데이터가 없는 시기에 무엇을 붙여야 하고, 학습형 TSAD 딥러닝이 데이터 몇 %부터 경량 통계를 이기는지를 숫자로 잰다. 계획서 전문은 [docs/plan_v4.md](docs/plan_v4.md)에 있다.

이 저장소에는 — 데이터 분할과 GDN 실행 — 의 코드·검증 기록과, 팀이 함께 쓰는 문서(계획서, 결정 로그, 점수 인터페이스 명세, 역할별 부탁문)가 들어 있다. 모델이 내놓는 유일한 산출물은 시점별 이상 점수 배열 `.npy`이고, 채점은 지우, 통계는 주혜가 분리해 맡는다. 두 환경은 점수 파일로만 통신한다.

## 읽는 순서

1. [docs/plan_v4.md](docs/plan_v4.md) — 무엇을 왜 재는지. 모든 절 번호 인용(7-1 등)의 원본
2. [DECISIONS.md](DECISIONS.md) — 결정 19건+의 장부. 값 하나하나가 어디서 왔는지 여기서 추적된다
3. [docs/score_interface.md](docs/score_interface.md) — 제 점수 파일을 받아 쓸 때 필요한 전부 (지우·주혜 필독)
4. 각자 역할 문서 — [docs/role_A.md](docs/role_A.md)(강혁), [docs/role_B.md](docs/role_B.md)(지우), [docs/role_C.md](docs/role_C.md)(주혜)

GraGOD 코드를 왜 패치해서 쓰는지 궁금하면 [experiments/exp00_gragod_recon/](experiments/exp00_gragod_recon/)의 검증 기록 다섯 편(RECON → VERIFICATION → SHAPEFLOW → PATCH_REVERIFY → ORCHESTRATION)에 근거가 줄 번호 단위로 남아 있다. 요지만 말하면, 포팅 구현의 입력 변환과 smoothing에서 버그 2건을 실행으로 실증했고, 두 줄 패치와 후처리 자체 구현으로 해결했다.

## 폴더 안내

- `src/common/` — 파일명 규약(naming), smoothing, 정규화, 점수 저장. naming.py는 순수 stdlib라 채점기 쪽에서 복사해 써도 된다
- `src/data_split/` — 앞자르기·뒷자르기·validation 분할 (계획서 7-1·7-2의 구현)
- `src/gdn_runner/` — GDN 단일 실행 러너와 인접행렬 추출
- `configs/` — 하이퍼파라미터·점수 파이프라인·고정 환경. 값은 코드가 아니라 여기가 소유한다
- `experiments/` — exp00(검증 기록), exp01(분할 검증), exp02(GHL), exp03(HAI 시드 10개)
- `patches/gdn_input_transform.diff` — GraGOD 패치 diff. 베이스 ec8cd452에 이 diff를 적용하면 포크가 재현된다(D-14)
- `tests/` — 단위 테스트 29개. 루트에서 `python -m unittest discover -s tests`

## 환경

GDN 실행 환경은 [configs/environment.yaml](configs/environment.yaml)에 고정돼 있다(python 3.10, torch 2.2.2 등). 채점·통계 쪽 TSB-AD 환경과는 합치지 않는다 — 계획서 5-1의 환경 이원화 원칙이다.

## 규칙 하나만

7절 규칙과 사전 등록 항목은 개인 판단으로 바꾸지 않는다. 변경은 회의에서만, 결정은 DECISIONS.md에 남긴다.
