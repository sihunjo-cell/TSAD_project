# 시훈 GDN 잔여 작업 실행 계획

> **새 세션용 지시:** `AGENTS.md`와 이 문서를 먼저 읽고, `현재 시작점`의 한 단계만 수행한다. 구현에는 `superpowers:executing-plans`와 `ponytail`을 적용한다.

**목표:** 실제 실험 전까지 GDN 단일 러너, GHL·HAI 입력, 배치·완전성 검사와 HAI 인접행렬 분석 코드를 근거가 닫힌 상태로 준비한다.

**구조:** 공통 규약은 `src/`와 `configs/`가 소유하고 실험별 실행·산출물은 각 `experiments/expNN_*` 안에만 둔다. 단계마다 직전 산출물을 가볍게 확인하고, 해당 단계의 산출물 하나를 완성한 뒤 멈춘다.

**기술 환경:** Python 3.10.20, torch 2.2.2+cpu, torch-geometric 2.5.3, 패치된 GraGOD 포크 `485e26b0c6b1d63f4f3531c8d05597db82e9db29`.

**기준 문서:** `docs/plan_v4.md`, `DECISIONS.md`, `docs/manifest_draft.md`, `experiments/exp00_gragod_recon/ORCHESTRATION.md`.

## 공통 제약

- 실제 모델 학습은 4단계 전까지 실행하지 않는다. 3b에서는 단위 테스트와 합성 드라이런만 허용한다.
- GHL 원본은 `../shared_data/TSAD_project/TSB-AD-M`, HAI 23.05는 `../shared_data/TSAD_project/HAI-23.05`에 둔다. 실물 데이터는 저장소에 넣지 않는다.
- GHL 비율은 `5·10·20·50·100%`, HAI GDN 비율은 `10·100%`다.
- HAI는 86채널, `topk=22`다. GHL은 19채널, `topk=5`다.
- validation은 축소된 학습 구간 안에서 10%를 떼며 시계열·세션 경계를 넘는 윈도를 만들지 않는다.
- 정규화·smoothing·점수 파일 규약은 `AGENTS.md`와 D-04~D-17을 그대로 따른다.
- 근거가 충돌하면 임의로 고르지 않는다. 공식 파일 경로와 줄을 기록하고 `DECISIONS.md`에 새 결정을 남긴다.

## 현재 상태

| 단계 | 상태 | 산출물·근거 |
| --- | --- | --- |
| 0 | 완료 | 저장소 구조, `AGENTS.md`, `DECISIONS.md` |
| 1 | 완료 | `src/common/`, 파일명·저장·정규화·smoothing 테스트 |
| 2 | 완료 | `src/data_split/`, `experiments/exp01_split_check/` |
| 3a | 완료 | `experiments/exp00_gragod_recon/`의 RECON→PATCH_REVERIFY→ORCHESTRATION |
| 3b-0A | 완료 | `configs/data_preprocessing.yaml`, `docs/manifest_draft.md`, `experiments/exp01b_ghl_preflight/`, `experiments/exp01c_hai_preflight/`, D-20·D-21 |
| 3b-0B | 완료 | `configs/gdn_hyperparams.yaml`, `docs/gdn_hyperparameter_decisions.md`, D-22 |
| 3b-1 | 완료 | `src/gdn_runner/run_gdn_single.py`, `tests/test_run_gdn_single.py`, D-23 |
| 3b-2 | 완료 | `src/data_split/load_ghl_series.py`, `load_hai_sessions.py`, `tests/test_load_gdn_inputs.py`, D-24 |
| 3b-2R | 완료 | `src/gdn_runner/run_gdn_single.py`, `tests/test_run_gdn_single.py`, D-25 |
| 3b-3 | 완료 | exp02·exp03 배치·완전성, HAI 인접행렬·Jaccard 코드, D-26 |
| 3b-4 | 완료 | `dryrun_synthetic.py`, `dryrun_out/`, `tests/test_dryrun_synthetic.py`, D-27 |
| 3b-5 | 완료 | `docs/pre_run_checklist.md`, D-28~D-32, 차단점 3건 해제 |
| 3b-5R1 | 완료 | 실행 snapshot 전체 config 계약 보강, D-29 |
| 3b-5R2 | 완료 | GHL 뒷자르기 통제군 연결, D-30 |
| 3b-5R3 | 완료 | 3b 변경 commit·clean 상태·새 합성 드라이런 봉인, D-31·D-32 |
| 4 | 다음 시작점 | 사용자가 실험 실행을 명시한 뒤 GHL 1개 스모크부터 시작 |
| 5 이후 | 대기 | 앞 단계 완료 보고 전 착수 금지 |

## 3b-0B — 하이퍼파라미터 근거 봉인

**입력:** `configs/gdn_hyperparams.yaml`, RECON [G], 고정 GraGOD 포크의 `models/gdn/params*.yaml`, d-ailin/GDN의 `main.py`·`run.sh`·`train.py`, 계획서의 HAI seed 규격.

**산출물:** 실행 가능한 값만 남은 `configs/gdn_hyperparams.yaml`, 출처·충돌·선택 이유를 담은 `docs/gdn_hyperparameter_decisions.md`, `DECISIONS.md`의 새 결정 한 항목.

**판정 규칙:** 저자 레포끼리 값이 다르면 저데이터 비교에서 데이터 양 외 조건을 고정한다는 계획서 원칙을 우선한다. 근거 없이 탐색 범위를 새로 만들지 않는다. 최종 값마다 레포·파일·줄을 남긴다.

**최소 검증:** YAML 파싱, 미정 문자열 0개, 필수 키 존재, `n_epochs >= 2`, GHL `topk=5`, HAI `topk=22`, seed 규격 대조만 수행한다. 대형 CSV, 전체 테스트, 모델 학습은 실행하지 않는다.

**출구:** 설정·근거 문서·DECISIONS가 같은 값을 가리키면 완료다. 완료 보고 뒤 멈춘다.

## 3b-1 — 단일 러너 계약 고정

**입력:** 3b-0B 설정과 `experiments/exp00_gragod_recon/ORCHESTRATION.md`.

**수정 파일:** `src/gdn_runner/run_gdn_single.py`, 신규 `tests/test_run_gdn_single.py`.

**작업:** 외부 의존을 작은 경계에서 대체한 테스트로 validation 위치, 모델 인자 주입, trainnorm 통계 구간, score 길이·metadata, snapshot 내용을 먼저 고정한다. 러너는 분할 완료 배열만 받는 현재 책임을 유지한다.

**최소 검증:** `python -m unittest tests.test_run_gdn_single -v`와 `python -m compileall -q src/gdn_runner tests/test_run_gdn_single.py`만 실행한다.

**출구:** 합성 배열에서 학습 호출 경계와 산출 계약이 검증되면 멈춘다. 실제 GDN 학습은 금지한다.

## 3b-2 — GHL·HAI 입력 조립

**입력:** `docs/manifest_draft.md`, `configs/data_preprocessing.yaml`, 3b-1 러너 계약.

**생성 파일:** `src/data_split/load_ghl_series.py`, `src/data_split/load_hai_sessions.py`, `tests/test_load_gdn_inputs.py`.

**작업:** GHL은 파일명 `tr_` 경계로 한 시계열씩, HAI는 훈련 4세션을 세션별 앞자르기·validation한 뒤 한 모델 입력으로 조립한다. timestamp·label은 제외하고 세션 경계 정보를 함께 반환한다.

**최소 검증:** 작은 임시 CSV fixture로 `python -m unittest tests.test_load_gdn_inputs -v`만 실행한다. 원본 전체 스캔은 하지 않는다.

**출구:** 경계·열·비율·scaler fit 범위를 fixture로 확인하면 멈춘다.

## 3b-2R — 세션 경계 러너 연결

**입력:** 3b-1 러너 계약과 3b-2의 세션별 배열·D-24.

**수정 파일:** `src/gdn_runner/run_gdn_single.py`, `tests/test_run_gdn_single.py`.

**작업:** 세션별 `SlidingWindowDataset`을 만든 뒤 dataset만 합쳐 한 모델을 학습한다. raw tensor를 먼저 연결하지 않는다. trainnorm 통계도 세션별 오차를 계산한 뒤 합친다. HAI test 2세션은 모델을 다시 학습하지 않고 각각 점수·metadata를 저장한다.

**최소 검증:** 기존 합성 단일 세션 계약과 새 다중 세션 계약을 `python -m unittest tests.test_run_gdn_single -v`로 확인한다. 실제 GDN 학습은 금지한다.

**출구:** 합성 세션 길이별 window 수의 합이 맞고 경계 횡단 window가 0이며 test 2세션 산출물이 분리되면 멈춘다.

## 3b-3 — 배치·완전성·인접행렬 코드

**입력:** 3b-2R 러너와 3b-2 입력 조립.

**생성 파일:** `experiments/exp02_gdn_ghl/run_batch.py`, `experiments/exp02_gdn_ghl/check_completeness.py`, `experiments/exp03_gdn_hai_seed10/run_batch.py`, `experiments/exp03_gdn_hai_seed10/check_completeness.py`, `experiments/exp03_gdn_hai_seed10/compute_jaccard_agreement.py`.

**작업:** 결과가 있으면 건너뛰고 실패 조합은 `logs/failures.csv`에 남긴다. GHL은 `25×5×3`, HAI는 `2×10` 실행 명세만 만든다. HAI는 실행별 최종 TopK edge 집합을 저장하고 seed 10개의 45쌍 Jaccard를 계산한다. 채점·VUS-PR·δ 계산은 이 저장소에 만들지 않는다.

**최소 검증:** 임시 실험 폴더에서 조합 수, resume, 실패 기록, 누락 검출, 손계산 Jaccard만 단위 테스트한다. 모델 학습은 금지한다.

**출구:** 빈 점수 폴더에서 예상 누락 수, 가짜 완성 폴더에서 누락 0, Jaccard 손계산이 맞으면 멈춘다.

**구현 결정:** exp02는 `runs/series_XX/rRRR/sS/`, exp03은 `runs/rRRR/sS/`를 실행 단위로 쓴다. exp03의 self-edge 포함·제거 인접행렬은 `adjacency/`, Jaccard 쌍별 값과 요약은 `analysis/`에 둔다. 완전성 검사는 점수·metadata·best checkpoint·early stopping 로그·config 스냅숏을 모두 확인하며, exp03은 인접행렬 2벌도 확인한다. D-26에 근거를 적었다.

## 3b-4 — 합성 드라이런

**입력:** 3b-0B~3b-3 결과와 고정 `tsad_fixed` 환경.

**작업:** `experiments/exp00_gragod_recon/dryrun_synthetic.py`를 현재 계약에 맞춰 한 번 실행한다. 실제 GHL·HAI 파일은 읽지 않는다.

**최소 검증:** raw·smoothed × trainnorm·testnorm × 집계·채널별 8개 점수 배열, metadata, config·두 git hash snapshot, best checkpoint, early stopping 로그의 존재와 shape만 확인한다.

**출구:** 합성 산출물만으로 단일 실행 경로가 닫히면 멈춘다.

**실행 결과:** `tsad_fixed` Python 3.10.20에서 합성 학습을 한 번 실행했다. 집계 점수 4개 `(191,)`, 채널별 점수 4개 `(191, 5)`, metadata, 두 git hash snapshot, best checkpoint, early stopping 로그, TopK edge 복원이 모두 맞았다. 수치와 근거는 D-27에 기록했다. 실제 GHL·HAI 파일은 읽지 않았다.

## 3b-5 — 실험 전 준비 감사

**입력:** 3b 전체 산출물.

**산출물:** `docs/pre_run_checklist.md` 한 파일.

**작업:** 설정 미정값, 경로, 조합 수, seed, 파일명, snapshot, resume, 실패 로그, HAI 세션 경계, score-label offset을 점검한다. 실행 명령은 적되 실행하지 않는다.

**최소 검증:** 이 지점에서만 `python -m unittest discover -s tests -v`, YAML 파싱, `git diff --check`를 한 번 실행한다. 원본 데이터는 inventory와 snapshot만 대조한다.

**출구:** 체크리스트 전 항목에 파일 근거가 있고 미정값이 없으면 3b 완료다.

**감사 당시 결과:** 전체 테스트 56건과 YAML 4개 대조는 통과했고 B-01~B-03을 찾았다. 3b-5R1~R3에서 세 항목을 순서대로 해제했다. 하위 보강 단계에서는 해당 계약만 다시 검사했으며 전체 테스트를 반복하지 않았다.

## 3b-5R1 — 실행 snapshot 계약 보강

**입력:** `docs/pre_run_checklist.md` B-02와 현재 GHL·HAI 배치 config.

**수정 파일:** `src/gdn_runner/run_gdn_single.py`, GHL·HAI `run_batch.py`, 해당 계약 테스트.

**작업:** 실행 snapshot에 모델·학습 설정뿐 아니라 `data_preprocessing.yaml`, `scoring_pipeline.yaml` 전문, 입력 경로, feature 이름과 `session_splits`를 넣는다. 값은 기존 파일과 로더 반환값을 그대로 기록하며 새 설정을 만들지 않는다.

**최소 검증:** 실제 모델을 돌리지 않고 GHL 한 세션·HAI 다중 세션 fixture에서 snapshot JSON의 필수 키와 값만 검사한다.

**출구:** snapshot 하나만으로 실행 config와 데이터 분할 출처를 복원할 수 있으면 멈춘다. 3b-5R2의 뒷자르기는 시작하지 않는다.

**수행 결과:** runner가 실제로 읽은 두 YAML 전문과 batch가 넘긴 입력 경로·feature 이름·`session_splits`를 한 snapshot에 저장한다. GHL 단일 세션·HAI 다중 세션 fixture의 RED→GREEN과 관련 테스트 10건, 변경 파일 compile을 통과했다. B-02는 해제했으며 실제 모델은 실행하지 않았다. 근거는 D-29와 `docs/pre_run_checklist.md`에 있다.

## 3b-5R2 — GHL 뒷자르기 통제군 연결

계획서 7-1의 GHL `{100,20,5%}` 통제군을 주 실행과 섞이지 않는 경로에 연결한다. 100%가 앞·뒤에서 같은 배열이라는 기존 단위 테스트를 근거로 중복 실행 처리부터 D-30에 정한 뒤 로더·배치·완전성 테스트를 작성한다.

**수행 결과:** 5%·20%는 exp02의 `back_trim_runs/`와 `back_trim_logs/`에 분리했다. 100%는 splitter와 loader 출력이 모두 같아 주 실행 `runs/`에 한 번만 저장한다. 논리 조합 225개 가운데 추가 fit은 150개다. 관련 테스트 18건과 변경 파일 compile을 통과했으며 실제 모델은 실행하지 않았다. 근거는 D-30과 `docs/pre_run_checklist.md`에 있다.

## 3b-5R3 — 실험 전 봉인

B-01을 해제하도록 3b 변경을 commit하고 clean 상태를 확인한다. 새 commit hash로 합성 드라이런을 한 번 다시 통과시킨 뒤에만 4단계를 연다. GPU 본 배치를 쓸 경우 같은 버전의 CUDA 환경에서도 이 드라이런을 먼저 통과시킨다.

**수행 결과:** 3b 변경 범위에 원본 데이터와 모델 산출물이 섞이지 않았음을 확인하고 한 commit으로 봉인했다. 첫 드라이런은 외부 포크의 Git 소유권 판정 때문에 포크 hash가 `unknown(커밋 없음)`으로 저장돼 실패했다. 전역 설정 대신 명시된 저장소 경로에만 `safe.directory`를 적용하고 RED→GREEN 테스트로 고친 뒤 같은 봉인 commit에 포함했다. 최종 드라이런의 snapshot에서 TSAD hash는 실행 당시 HEAD와 같고 GraGOD hash는 고정값 `485e26b0c6b1d63f4f3531c8d05597db82e9db29`다. 점수 8개, metadata, best checkpoint, early stopping 로그, TopK 복원을 다시 확인했다. 동적 TSAD hash의 원본은 `dryrun_out/snapshots/config_snapshot.json`이며, 판정 근거는 D-31·D-32에 적었다. GHL·HAI 원본은 읽지 않았다.

## 4 — GHL 실행

사용자가 실험 실행을 명시한 세션에서만 시작한다. 시계열 1개·10%·seed 1 스모크 결과를 사람이 확인한 뒤 `25×5×3` 배치로 넓힌다. completeness 누락 0을 확인하고 멈춘다.

## 5 — HAI 실행

4단계 완료 뒤 시작한다. HAI `10·100% × seed 1~10`을 실행하고 점수 배열 20실행분, 인접행렬 20개, 조건별 seed 쌍 45개의 Jaccard 표를 남긴다. 채점 결과는 만들지 않는다.

## 새 세션 시작 절차

1. `AGENTS.md`, 이 문서, `DECISIONS.md`를 읽는다.
2. `현재 시작점` 바로 전 단계의 설정·로그·결정 기록만 가볍게 대조한다.
3. 현재 단계 하나만 수행한다.
4. 완료 보고에 완료 단계, 검증 명령과 결과, 설계 근거 파일·줄, 다음 단계를 적고 멈춘다.

현재 첫 명령은 “사용자가 실험 실행을 명시하면 D-31과 합성 드라이런 snapshot의 TSAD hash=현재 HEAD, 고정 포크 hash, clean 작업 트리를 가볍게 확인한 뒤 4단계의 GHL 시계열 1개·10%·seed 1 스모크만 수행하라”이다.
