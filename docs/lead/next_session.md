# 다음 세션 시작점

## 먼저 읽을 파일

1. `AGENTS.md`
2. `docs/lead/plan_v4.md`
3. `docs/lead/process_0_preverify.md`
4. 이 문서

현재 결정을 바꿀 때만 `docs/lead/decisions.md`를 찾는다. 역할별 입력은 `docs/role_A/`, `docs/role_B/`, `docs/role_C/`에 있다.

## 현재 게이트

계층 2의 파라미터 EDA와 CI-AE·LSTM-AD·USAD·GDN adapter, GHL·HAI runner를 완성했다.
합성 데이터로 optimizer 갱신, checkpoint 복원, 점수 8벌, metadata와 완료 판정을 확인했다.
HAI형 GDN도 train 4세션·test 2세션과 인접행렬 두 종류를 통과했다. 기존 EDA는 다시 만들지
않았고 실데이터 학습도 실행하지 않았다.

실데이터 runner는 현재 dirty 작업 트리에서 의도적으로 멈춘다. 외부 GraGOD fork와 삭제된
소문자 GDN 경로는 더 이상 필요하지 않다.

## 다음 작업

이번 변경을 commit해 작업 트리를 clean 상태로 만든다. 연구실 서버에서 accelerator와
`tsad_fixed` 환경을 확인하고 `python -m tests.checks.run_tier2_dryrun`을 다시 통과한 뒤에만
GHL·HAI 본실험 runner를 시작한다.

지우의 채점기와 주혜의 난이도·통계 산출물은 모델 구현 종료 조건이 아니라 최종 통합 실험
조건이다. 도착 전에는 VUS-PR 계산과 난이도·통계 집계만 시작하지 않는다.
