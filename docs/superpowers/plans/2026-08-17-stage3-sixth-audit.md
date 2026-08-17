# 3단계 6차 논리 감사 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 3a·3b의 확정 계약이 실제 실행 API와 완전성 판정까지 손실 없이 이어지는지 다시 확인하고, 재현되는 결함만 고친다.

**Architecture:** D-01~D-42와 YAML을 원천 계약으로 두고 설정→로더→학습·예측→점수→snapshot→완료 판정 경로를 정방향과 재개 방향에서 각각 추적한다. 5차까지 확인한 수치와 원본 CSV는 다시 계산하지 않는다.

**Tech Stack:** Python 3.10.20, 표준 라이브러리, PyYAML, NumPy, 기존 `unittest` 모음.

**Spec:** `docs/NEXT_SESSION_PLAN.md`, `docs/pre_run_checklist.md`, `DECISIONS.md` D-01~D-42.

## Global Constraints

- 실제 GHL·HAI 학습, 합성 모델 실행, 원본 CSV 스캔과 SHA-256 재계산은 하지 않는다.
- 비율·seed·validation·window·topk와 D-34의 `L-W` 정렬은 새 반례가 없으면 바꾸지 않는다.
- 새 결함은 현재 허용 입력이나 중단·재개 상태에서 재현해야 한다.
- 수정이 필요하면 실패 테스트→최소 구현→관련 테스트 순서로 진행한다.
- commit하지 않고 4단계로 넘어가지 않는다.

---

### Task 1: 계약과 현재 구현 매핑

**Files:**
- Read: `configs/*.yaml`
- Read: `src/**/*.py`
- Read: `experiments/exp00_*`~`experiments/exp03_*`
- Read: `tests/test_*.py`

**Interfaces:**
- Consumes: D-01~D-42와 5차 감사 결과.
- Produces: 설정값마다 생성자와 실제 소비자, 테스트를 잇는 매핑.

- [x] **Step 1:** `rg --files`와 `rg`로 3a·3b 코드·설정·테스트의 전체 목록과 하드코딩 값을 수집한다.
- [x] **Step 2:** 5차 변경 파일과 현재 diff를 대조해 감사 기준선이 달라지지 않았는지 확인한다.
- [x] **Step 3:** 값마다 YAML 원본, 로더·러너 소비 지점, 단위 테스트를 연결한다.

### Task 2: 실행 API 정방향 추적

**Files:**
- Read: `src/gdn_runner/run_gdn_single.py`
- Read: `src/data_split/*.py`
- Read: `src/common/*.py`
- Read: `../gragod-fork/datasets/dataset.py`
- Read: `../gragod-fork/models/train.py`
- Read: `../gragod-fork/models/predict.py`
- Read: `../gragod-fork/gragod/training/*.py`

**Interfaces:**
- Consumes: Task 1의 계약 매핑.
- Produces: 입력 shape·dtype·device, 학습 인자, 예측 정렬, 점수·snapshot의 실제 전달 판정.

- [x] **Step 1:** loader 반환값이 GraGOD dataset과 model 호출까지 어떤 shape·dtype으로 전달되는지 추적한다.
- [x] **Step 2:** YAML의 모든 고정 학습값이 실제 trainer·optimizer·scheduler·early stopping에 도달하는지 대조한다.
- [x] **Step 3:** trainnorm·testnorm, smoothing, 집계, 파일명, metadata가 D-04~D-16·D-34와 같은지 확인한다.

### Task 3: 재개·대조군 역방향 추적

**Files:**
- Read: `experiments/exp02_gdn_ghl/*.py`
- Read: `experiments/exp03_gdn_hai_seed10/*.py`
- Read: `src/common/run_completion.py`
- Read: `tests/test_gdn_batch.py`

**Interfaces:**
- Consumes: 실제 산출물 계약과 실험 조합.
- Produces: 완료→재개→재사용 경로가 올바른 조합과 신원을 가리키는지에 대한 반례 목록.

- [x] **Step 1:** 주 실행·back-trim·−TOPK·TopK 민감도·HAI 조합을 각 완전성 함수에서 역추적한다.
- [x] **Step 2:** 빈 파일, 깨진 JSON·NPY, 부분 산출물, 이전 commit, 중복 spec, 실패 후 재개를 대입한다.
- [x] **Step 3:** 현재 테스트가 못 잡는 실제 반례만 수정 후보로 남긴다.

### Task 4: 재현 결함 수정과 기록

**Files:**
- Modify only if needed: 재현 원인을 소유한 기존 파일 한 곳.
- Test only if needed: 가장 가까운 기존 `tests/test_*.py`.
- Modify: `DECISIONS.md`
- Modify: `docs/pre_run_checklist.md`
- Modify: `docs/NEXT_SESSION_PLAN.md`
- Modify: `docs/superpowers/plans/2026-08-17-stage3-sixth-audit.md`

**Interfaces:**
- Consumes: Task 2·3에서 재현된 반례.
- Produces: 실패 테스트와 최소 수정, 또는 코드 변경이 필요 없다는 근거 기록.

- [x] **Step 1:** 실행 코드에서 재현되는 새 결함이 없음을 확인해 실패 테스트 추가를 생략한다.
- [x] **Step 2:** 실행 코드는 바꾸지 않고, Manifest 확정 전 범위가 남은 계획서 세 문장만 고친다.
- [x] **Step 3:** 고친 문서와 유지한 실행 결정을 D-43과 세 작업 문서에 같은 뜻으로 기록한다.

### Task 5: 출구 검증

**Files:**
- Verify: 변경 코드와 관련 테스트.
- Verify: `configs/*.yaml`, 고정 GraGOD 포크 상태, `git diff --check`.

**Interfaces:**
- Consumes: Task 4 결과.
- Produces: 6차 감사 완료 증거와 다음 시작점.

- [x] **Step 1:** 수정 전 고정 환경 전체 단위 테스트 104건을 통과했고, 문서 수정 뒤 경로·핵심 수치만 대조한다.
- [x] **Step 2:** YAML·코드가 바뀌지 않았으므로 중복 compile은 생략하고 diff와 고정 포크 상태를 확인한다.
- [x] **Step 3:** 모델·원본 데이터 산출물이 생기지 않았는지 확인하고 4단계로 넘어가지 않은 채 멈춘다.

## 수행 결과

- D-34~D-42와 YAML에서 loader, GraGOD 생성자·trainer, 예측, 후처리, snapshot, 완료 판정까지 양방향으로 추적했다. 비율·seed·validation·window·topk, `L-W` 정렬, 실험 조합 수는 바꿀 근거가 없었다.
- 주 실행·back-trim·−TOPK·TopK 민감도·HAI의 재개 경로는 현재 commit 신원과 최종 `COMPLETE` 표식을 함께 요구한다. 빈 파일, 부분 산출물, 형식이 깨진 snapshot, 이전 commit은 미완료로 돌아간다. 중복 spec은 YAML 생성 경로에서 나오지 않으며, 실패한 실행은 표식이 없어 다시 돈다.
- `COMPLETE` 뒤 외부에서 파일을 변조하거나 비트가 깨지는 경우까지 매번 배열 전체를 다시 읽는 것은 D-37의 완료 판정 범위가 아니다. 현재 실행 경로가 깨진 JSON·NPY를 성공 산출물로 만드는 반례는 재현되지 않았고, 검증 비용 규율에 따라 출력 전체 해시 검사는 추가하지 않았다.
- 실행 코드의 새 결함은 없었다. 다만 `docs/plan_v4.md` 세 곳에 Manifest 확정 전 HAI 범위 `59~86ch`가 남아 있었다. HAI 23.05의 확정값 `86ch`로 바로잡았으며 `topk=22`와 PCA 30 결정은 그대로다.
- 수정 전 고정 `tsad_fixed` 환경 전체 단위 테스트 104건이 통과했다. 실제 GHL·HAI·합성 모델, 원본 CSV·SHA-256 재검사, commit은 수행하지 않았다.
