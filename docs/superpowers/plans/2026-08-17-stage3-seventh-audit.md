# 3단계 7차 최종 논리 감사 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 4단계 전 마지막 정적 감사로 3a·3b의 근거, 통계 계약, 실행 봉인, 중단·재개 경계를 독립적으로 다시 확인하고 재현되는 결함만 고친다.

**Architecture:** D-01~D-43과 현재 YAML을 원천 계약으로 둔다. 6차에서 끝낸 설정 전달 대조를 그대로 반복하지 않고 근거 참조→데이터 누수→산출물 원자성→재개·재사용 순서로 역추적한다. 현재 미커밋 누적 변경이 감사 대상이므로 별도 worktree를 만들지 않는다.

**Tech Stack:** Python 3.10.20, 표준 라이브러리, PyYAML, NumPy, 기존 `unittest` 모음.

**Spec:** `AGENTS.md`, `docs/NEXT_SESSION_PLAN.md`, `docs/pre_run_checklist.md`, `DECISIONS.md` D-01~D-43.

## Global Constraints

- 실제 GHL·HAI 학습, 합성 모델 실행, 원본 CSV 스캔과 SHA-256 재계산은 하지 않는다.
- 비율·seed·validation·window·topk와 `L-W` 정렬은 새 반례가 없으면 바꾸지 않는다.
- 검사 명령 오류는 산출물 결함과 구분한다.
- 코드 결함이 재현되면 systematic-debugging과 TDD 순서로 최소 수정한다.
- commit하지 않고 4단계로 넘어가지 않는다.

---

### Task 1: 6차 기준선과 근거 참조 감사

**Files:**
- Read: `AGENTS.md`
- Read: `DECISIONS.md`
- Read: `docs/*.md`
- Read: `docs/superpowers/plans/*.md`
- Read: `patches/gdn_input_transform.diff`
- Read: `experiments/exp00_gragod_recon/PATCH_REVERIFY.md`

**Interfaces:**
- Consumes: D-01~D-43, 6차 감사 결과, 현재 dirty diff.
- Produces: 존재하지 않는 경로·낡은 줄 번호·서로 충돌하는 현재 계약 목록.

- [x] **Step 1:** `rg`로 고정 hash, `L-W`, HAI 86채널, 비율·topk, 외부 레포 경로를 문서 전체에서 대조한다.
- [x] **Step 2:** 현재 계약이 인용한 고정 GraGOD 파일과 패치 재현 파일이 실제로 존재하는지 확인한다.
- [x] **Step 3:** 과거 기록은 상태 문구로 격리됐는지, 현재 실행 지시와 섞이는 참조만 남았는지 판정한다.

### Task 2: 데이터 누수와 표본 단위 감사

**Files:**
- Read: `src/data_split/*.py`
- Read: `src/gdn_runner/run_gdn_single.py`
- Read: `configs/data_preprocessing.yaml`
- Read: `configs/scoring_pipeline.yaml`
- Read: `tests/test_load_gdn_inputs.py`
- Read: `tests/test_run_gdn_single.py`

**Interfaces:**
- Consumes: GHL 파일 경계, HAI 4+2 세션 계약, 비율별 앞자르기·validation 규칙.
- Produces: train·validation·test 사이의 값·통계·window 누수 반례 판정.

- [x] **Step 1:** 원본 경계→비율 절단→validation→scaler fit·transform 순서를 두 loader에서 다시 추적한다.
- [x] **Step 2:** HAI 세션을 합치는 지점이 scaler 통계와 dataset 결합뿐인지 확인하고 raw window가 세션을 넘지 않는지 판정한다.
- [x] **Step 3:** trainnorm·testnorm과 라벨 정렬에서 test 정보가 본표 학습·정규화로 역류하는 호출이 없는지 찾는다.

### Task 3: 실행 봉인과 산출물 원자성 감사

**Files:**
- Read: `.gitignore`
- Read: `src/common/verify_run_context.py`
- Read: `src/common/verify_input_files.py`
- Read: `src/common/run_completion.py`
- Read: `src/common/save_scores.py`
- Read: `experiments/exp02_gdn_ghl/*.py`
- Read: `experiments/exp03_gdn_hai_seed10/*.py`

**Interfaces:**
- Consumes: snapshot 선행 저장, 종료 Git 검증, `COMPLETE` 최종 표식 계약.
- Produces: 실패·중단·재시작 때 완료 오인이나 서로 다른 실행의 산출물 혼합 가능성 판정.

- [x] **Step 1:** 실행 산출물 경로가 Git clean 검사를 깨지 않도록 `.gitignore`에 들어 있는지 대조한다.
- [x] **Step 2:** snapshot 작성 전후, 모델 성공, 점수·HAI graph 저장, 종료 검증, `COMPLETE` 순서를 따라 예외 위치마다 재개 결과를 판정한다.
- [x] **Step 3:** 주 실행 재사용 조건인 back-trim 100%와 TopK k=5가 경로뿐 아니라 입력·모델 계약도 같은지 확인한다.

### Task 4: 수치·통계·검정 설계 일관성 감사

**Files:**
- Read: `docs/plan_v4.md`
- Read: `docs/gdn_hyperparameter_decisions.md`
- Read: `experiments/exp01b_ghl_preflight/ANALYSIS.md`
- Read: `experiments/exp01c_hai_preflight/ANALYSIS.md`
- Read: `configs/*.yaml`

**Interfaces:**
- Consumes: EDA 봉인값과 계획서의 실험 질문·대응 검정 단위.
- Produces: 임의 파라미터, 표본 독립성 과장, 비율 역할과 실행 격자의 충돌 여부.

- [x] **Step 1:** 5%·10%·20%·50%·100% 역할과 HAI 10%·100% 축약 근거를 EDA 판정문과 연결한다.
- [x] **Step 2:** GHL 25개를 대응 표본으로 쓰는 통계 해석이 같은 시뮬레이터라는 한계를 명시하는지 확인한다.
- [x] **Step 3:** topk 5·22와 민감도 2·5·10, seed 3·10, validation 0.1에 결과를 본 뒤 바꿀 여지가 남아 있지 않은지 확인한다.

### Task 5: 재현 결함 수정과 최종 기록

**Files:**
- Modify only if needed: 재현 원인을 소유한 기존 파일.
- Test only if needed: 가장 가까운 기존 `tests/test_*.py`.
- Modify: `DECISIONS.md`
- Modify: `docs/pre_run_checklist.md`
- Modify: `docs/NEXT_SESSION_PLAN.md`
- Modify: `docs/superpowers/plans/2026-08-17-stage3-seventh-audit.md`

**Interfaces:**
- Consumes: Task 1~4의 재현 반례.
- Produces: 최소 수정과 회귀 증거, 또는 코드 수정 불필요 판정.

- [x] **Step 1:** 실행 코드 결함은 재현되지 않았다. Jaccard 경로는 수정 전 `NOT_IGNORED`를 실패 증거로 고정했다.
- [x] **Step 2:** 원인을 소유한 `.gitignore`, preflight 생성기, 근거·통계 문서만 최소 수정하고 GHL preflight 관련 테스트를 통과시켰다.
- [x] **Step 3:** D-44와 세 작업 문서에 수정값, 유지값, 남은 한계를 같은 뜻으로 기록한다.

### Task 6: 최소 출구 검증

**Files:**
- Verify: 이번 변경 파일.
- Verify: `configs/*.yaml`, 고정 GraGOD 포크 상태, `git diff --check`.

**Interfaces:**
- Consumes: Task 5 결과.
- Produces: 7차 감사 완료 증거와 봉인 단계 진입점.

- [x] **Step 1:** GHL preflight 관련 단위 테스트 7건과 경로·수치·참조 대조를 통과시켰다.
- [x] **Step 2:** 실행 논리를 바꾸지 않아 전체 테스트와 compile은 반복하지 않았다.
- [x] **Step 3:** 고정 포크 hash·clean 상태와 diff 공백 오류를 확인하고 4단계 전에서 멈춘다.

## 감사 결과

- 유지: GHL `5·10·20·50·100%`, HAI `10·100%`, seed `3·10`, validation `0.1`, W=5, topk `5·22`, 점수 길이 `L-W`.
- 수정: 패치 재검증 문서의 정확한 경로, exp03 Jaccard 산출물의 Git 제외, GHL preflight의 D-22 현재 상태와 재생성 문구.
- 해석 제한: GHL 25개로 계산한 Wilcoxon·TOST·bootstrap은 benchmark 내부 요약이며 제조 공정 모집단 추론이 아니다.
- 미실행: 모델 학습, 합성 드라이런, 원본 CSV·SHA-256 재검사, commit.
