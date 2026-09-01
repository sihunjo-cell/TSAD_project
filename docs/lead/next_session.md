# 다음 세션 시작점

## 현재 게이트

[계획서 v5](plan_v5.md)의 0·1단계를 마쳤고 2단계 TSB 비-GHL 18개 튜닝을 마무리하는 중이다.
저장소에 마지막으로 기록한 원격 상태는 primary 물리 실행 1,170건 중 1,136건 완료였지만,
이후 Lightning 실행 상태는 로컬에서 추측하지 않는다. CPU 채점 전에 원격 manifest 1,224행과
primary complete 1,170행을 확인해야 한다. 이 수가 맞으면 모델 실행은 끝난 것이며 채점과 선택만
남는다.

Dev18 입력 감사, 3,276행 정적 feasibility, 시계열별 `ℓ_max`, VUS-PR evaluator와 단일 실행
진입점은 봉인했다. 완료한 물리 score는 지우거나 다시 계산하지 않는다. primary logical score
원표 1,602행이 완성돼야 선택 단계로 넘어간다.

## 다음에 할 일 하나

Lightning에서 manifest 1,224행과 primary 1,170행을 먼저 확인한다. 통과하면 고코어 CPU로 바꾸고
[모델 완료 후 고코어 CPU에서 채점](lightning_studio.md#모델-완료-후-고코어-cpu에서-채점) 절차만
실행한다. 별도 진입점은 모델 runner를 호출하지 않으며, 물리 score별 원자적 checkpoint로 끊긴
지점부터 재개한다. 최종 완료는 commit·manifest·모든 결과 SHA를 묶은 영수증으로 확인한다.
`reset_lightning_dev18.py`, `run_lightning_dev18.py`와 일반
`run_dev18_tuning.py`는 실행하지 않는다.

manifest가 1,224행 또는 primary가 1,170행보다 적으면 CPU 채점을 시작하지 않는다. 그때만 기존
L4 복구 절차로 돌아가 남은 모델 실행을 먼저 끝낸다. CPU 채점 경로는 GPU runtime과 checkpoint
smoke 신원만 요구하지 않으며, budget·registry·feasibility·`ℓ_max`·evaluator 봉인, 입력 CSV와
score·metadata SHA-256, label 정렬은 그대로 검증한다.

## 봉인 상태

- Dev18 감사: `approved_with_disclosed_source_training_contamination`
- feasibility: 3,276행 중 feasible 2,592행, structurally infeasible 684행
- equal-trial config 수: Tier 1·2·3 순서로 `1·2·1`
- budget: `b5367ad431093` 유지, CPU 채점은 primary 1,170건 complete 확인 뒤에만 허용
- VUS-PR: 공식 TSB-AD `opt` 구현과 대조 통과, 최대 절대 오차 0
- `ℓ_max`: 18개 시계열별 training-only snapshot 봉인
- checkpoint: TimeRCD·TSPulse Dev18 1,536×2 forward와 TSPulse batch 동등성 통과
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
