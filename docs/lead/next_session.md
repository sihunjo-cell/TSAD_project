# 다음 세션 시작점

## 현재 게이트

[계획서 v5](plan_v5.md)의 0·1단계를 마쳤다. 지금은 2단계 TSB 비-GHL 18개 튜닝을 실행하기
직전이다. 전체 HPO와 채점은 아직 시작하지 않았다. 다만 현재 project commit에서 새 gate 증거가
없어 exact panel은 blocked다.

Dev18 입력 감사, 3,276행 정적 feasibility, 시계열별 `ℓ_max`, VUS-PR evaluator와 단일 실행
진입점은 봉인했다. exact panel은 시계열마다 물리 실행 65건이고 18개 전체로는 1,170건이다.
primary logical score 원표는 1,602행이 완성돼야 선택 단계로 넘어간다. 기존 `c...` config 행과
`budget_id=b5367ad431093`은 새 TimeRCD smoke와 TSPulse 동등성이 같을 때만 유지한다.

## 다음에 할 일 하나

Lightning AI의 non-interruptible L4에서 [새 commit 시작 절차](lightning_studio.md#새-commit에서-한-번만-초기화하고-시작)를 처음부터
수행한다. fresh checkpoint smoke, TSPulse batch 1 대 batch 32 동등성, 80% 자원 JSON이 모두
`passed`가 되기 전에는 `run_lightning_dev18.py`를 실행하지 않는다. adaptive 정책과 모델별 수동
실행은 이 gate의 fallback이 아니다.

기본 device는 CUDA다. CUDA가 활성화되지 않은 로컬 노트북에서는 CPU로 되돌리지 않고 즉시
중단한다. 원격 CPU에서 실행할 때만 `--remote-execution --device cpu`를 명시한다. 실행 전 조건만
다시 볼 때는 `--prepare`, 160×2 실데이터 gate만 확인할 때는 `--smoke`를 쓰며 smoke도 CUDA 또는
원격 환경에서만 허용한다. 둘 다 HPO를 시작하지 않는다.

## 봉인 상태

- Dev18 감사: `approved_with_disclosed_source_training_contamination`
- feasibility: 3,276행 중 feasible 2,592행, structurally infeasible 684행
- equal-trial config 수: Tier 1·2·3 순서로 `1·2·1`
- budget: 새 gate 동등성 전까지 조건부 `sealed`, 실행 `blocked`
- VUS-PR: 공식 TSB-AD `opt` 구현과 대조 통과, 최대 절대 오차 0
- `ℓ_max`: 18개 시계열별 training-only snapshot 봉인
- checkpoint: 과거 TimeRCD·TSPulse Dev18 1,536×2 forward는 현재 commit에서 재실행 필요
- Python: `tsad_models_311`, CPython 3.11.14
- 실데이터 gate: MWVAR 160×2 smoke 통과

독립 oracle은 Dev18 실행의 별도 하드 게이트가 아니다. 봉인 commit의 공식 TSB-AD 구현과 현재
evaluator가 같은 입력에서 일치하고 evaluator SHA가 보고서에 묶였으므로 이 근거로 충분하다.
새 commit의 checkpoint·동등성·L4 자원 증거는 별도 실행 gate다.

## 범위 구분

ALoRa는 구현 미완료가 아니라 Dev18 18개 전체에 공통으로 적용할 config가 없어
`unavailable`이다. 억지로 panel에 넣지 않는다. PCA_LEGACY는 `q100`, PaAno는 `q40` 이상,
GDN은 `q10` 이상이라는 봉인된 지원 범위를 유지한다. TSPulse의 주 variant는 `raw_max`이며
`time·fft·pred`는 진단용이다.

GHL25·HAI의 Role-A 최종 인수는 다음 본실험 게이트다. 아직 남아 있지만 Dev18 튜닝을 막지 않는다.

## 완료 보고 형식

현재 게이트, 실행한 panel 수, 실패·재시도 상태, 1,602행 원표 완성 여부, 선택 결과와 결과 폴더만
보고한다. 파일별 조사 과정은 나열하지 않는다.
