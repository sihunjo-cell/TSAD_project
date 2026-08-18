# 다음 세션 시작점

## 먼저 읽을 파일

1. `AGENTS.md`
2. `docs/lead/plan_v4.md`
3. `docs/lead/process_0_preverify.md`
4. 이 문서

현재 결정을 바꿀 때만 `docs/lead/decisions.md`를 찾는다. 역할별 입력은 `docs/role_A/`, `docs/role_B/`, `docs/role_C/`에 있다.

## 현재 게이트

실데이터 학습은 금지다. GDN 실행 흐름은 준비됐지만 재현 기준·train stride·topk는 아직
잠정값이다. 나머지 계층 1·2·3 모델도 구현되지 않았고, 지우의 채점 규칙과 주혜의
난이도·통계 규칙도 대기 중이다.

## 다음 작업

제가 담당하는 모델을 한 종류씩 준비한다. 원 논문, 공식 구현, TSB-AD wrapper를 실제로 대조해 입력·출력과 고정값을 기록한 뒤 최소 구현과 합성 단위 테스트까지만 만든다. 모든 모델은 `docs/role_B/score_interface.md`의 점수·metadata 형식을 따른다.

외부 산출물이 먼저 오면 `docs/lead/process_0_preverify.md`의 실행 전 조건에 적힌 파일만 갱신한다. 실제 GHL·HAI 배치, VUS-PR 계산, 난이도 분할, Jaccard와 통계 집계는 시작하지 않는다.
