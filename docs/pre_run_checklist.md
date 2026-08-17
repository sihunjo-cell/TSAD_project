# GDN 실험 전 준비 감사

- 감사일: 2026-08-16
- 보강 확인일: 2026-08-17
- 봉인 확인일: 2026-08-17
- 범위: 3b-0A부터 3b-4까지의 GDN 준비 산출물
- 원칙: GHL·HAI 원본은 다시 읽지 않았다. 확정 manifest, preflight inventory·snapshot, 합성 드라이런 산출물만 대조했다.
- 착수 판정: 3b 준비 완료. B-01~B-03을 모두 해제했다. 4단계는 사용자가 실험 실행을 명시한 뒤 시작한다.

## 차단 항목 해제 기록

| 번호 | 차단 사유 | 확인 근거 | 해제 조건 |
| --- | --- | --- | --- |
| B-01 (해제) | 3b 변경 범위에서 원본 데이터와 모델 산출물을 제외하고 한 commit으로 봉인했다. 작업 트리가 clean인 새 HEAD에서 합성 드라이런을 다시 실행했으며 snapshot의 TSAD hash가 그 HEAD와 같다. | `git status --short`; `experiments/exp00_gragod_recon/dryrun_out/snapshots/config_snapshot.json`; D-31 | commit 직후와 드라이런 직후 `git status --short` 출력이 비었고 합성 검사항목 a~g가 모두 통과했다. |
| B-02 (해제) | snapshot에 `data_preprocessing.yaml`·`scoring_pipeline.yaml` 전문과 입력 경로·feature 이름·`session_splits`를 함께 저장한다. 두 batch는 로더 반환값을 그대로 넘긴다. | `src/gdn_runner/run_gdn_single.py:275-278,329-338`; GHL `run_batch.py:94-106`; HAI `run_batch.py:100-112`; `tests/test_run_gdn_single.py:228-280,301-368`; `tests/test_gdn_batch.py:91-180`; D-29 | 집중 테스트에서 GHL 단일 세션과 HAI 다중 세션의 전달값·snapshot JSON을 대조했다. |
| B-03 (해제) | GHL `{5,20,100%}` 뒷자르기를 loader·batch·완전성 검사에 연결했다. 5%·20%는 `back_trim_runs/`, 100%는 동일한 주 실행 `runs/`를 쓴다. 실패 로그는 `back_trim_logs/`에 둔다. | `docs/plan_v4.md:213`; `src/data_split/load_ghl_series.py:20-93`; exp02 `check_completeness.py:14-106`, `run_batch.py:48-151`; `tests/test_back_trim_split.py:30-40`; `tests/test_load_gdn_inputs.py:89-149`; `tests/test_gdn_batch.py:142-250`; D-30 | 관련 테스트에서 분할 범위·train-only scaler·225개 조합·100% 공유·resume·실패 로그를 확인했다. |

현재 로컬 `tsad_fixed`는 `torch 2.2.2+cpu`라 전체 395개 주 실행을 GPU 시간 추정치와 곧바로 비교할 수 없다. GPU에서 본 배치를 돌릴 경우 같은 버전의 CUDA 환경에서 합성 드라이런을 먼저 통과시킨다.

## 통과 항목

| 항목 | 판정과 확인값 | 파일 근거 |
| --- | --- | --- |
| 설정 미정값 | 통과. 실행 YAML에 TBD·TODO·Manifest 대기 문자열이 없다. `label_aggregation: null`은 `downsample_factor: 1`이라 집계하지 않는다는 확정값이다. | `configs/environment.yaml`; `configs/gdn_hyperparams.yaml:4-32`; `configs/data_preprocessing.yaml:4-29`; `configs/scoring_pipeline.yaml:4-28`; D-20~D-22 |
| 고정 환경 | 통과. 실측값은 Python 3.10.20, numpy 1.26.4, torch 2.2.2+cpu, torch-geometric 2.5.3, pytorch-lightning 2.6.2, tensorboardX 2.6.2.2다. | `configs/environment.yaml`; `experiments/exp00_gragod_recon/PATCH_REVERIFY.md` 1절 |
| GraGOD 포크 | 통과. 경로 `../gragod-fork`, branch `fix/gdn-input-transform`, HEAD `485e26b0c6b1d63f4f3531c8d05597db82e9db29`, 작업 트리 clean이다. | `experiments/exp00_gragod_recon/PATCH_REVERIFY.md`; `patches/gdn_input_transform.diff`; D-14 |
| 데이터 위치·버전 | 통과. GHL·HAI 공유 경로와 포크·고정 Python이 존재한다. GHL inventory는 25행·19채널, HAI 훈련 inventory는 4세션·86채널이다. HAI 23.05 test 2세션과 label 2파일은 LFS 포인터가 아닌 CSV로 검수돼 있다. | `docs/manifest_draft.md:7-72`; `experiments/exp01b_ghl_preflight/logs/inventory.csv`; `experiments/exp01c_hai_preflight/logs/inventory.csv`; 두 preflight `snapshots/config_snapshot.json` |
| 비율·validation | 통과. 주 설정은 GHL `5·10·20·50·100%`, HAI `10·100%`다. GHL 뒷자르기는 `5·20·100%`만 허용한다. validation 0.1은 줄인 학습 구간의 뒤쪽에서 뗀다. | `configs/data_preprocessing.yaml:18-29`; `configs/gdn_hyperparams.yaml:26`; `src/data_split/load_ghl_series.py:20-69`; 계획서 7-1·7-2 |
| 전처리 | 통과. downsample 1, 초기 절단 0, timestamp·label 제외, train 부분에만 MinMaxScaler를 fit한다. HAI는 네 train 세션에 `partial_fit`하고 세션 경계를 합치지 않는다. | `configs/data_preprocessing.yaml:4-29`; `src/data_split/load_ghl_series.py:23-84`; `src/data_split/load_hai_sessions.py:71-133`; D-20·D-24 |
| 모델 설정 | 통과. window 5, embedding 64, GHL·HAI topk 5·22, batch 32, 최대 50 epoch, early stopping patience 10, seed GHL 1~3·HAI 1~10이 설정·결정 기록과 같다. | `configs/gdn_hyperparams.yaml:4-32`; `docs/gdn_hyperparameter_decisions.md`; D-22 |
| 주 실행 조합 | 통과. GHL은 `25×5×3=375`, HAI는 `2비율×10seed=20`번 fit한다. HAI 한 fit이 test 2세션을 함께 채점한다. | `experiments/exp02_gdn_ghl/check_completeness.py:14-19`; `experiments/exp03_gdn_hai_seed10/check_completeness.py:14-19`; D-25·D-26 |
| GHL 뒷자르기 통제군 | 통과. 논리 조합은 `25×3×3=225`개다. 100% 75개는 주 실행과 입력이 같아 한 번만 저장하므로 실제 추가 fit은 `25×2×3=150`개다. 추가 점수 배열은 1,200개, metadata는 150개다. | exp02 `check_completeness.py:14-39`; `tests/test_back_trim_split.py:30-40`; `tests/test_load_gdn_inputs.py:134-149`; D-30 |
| 점수 파일명·개수 | 통과. fit·test 세션마다 raw/smoothed × trainnorm/testnorm × 집계/채널별 8개다. 주 실행 전체는 GHL 3,000개, HAI 320개 점수 배열과 metadata 375·40개다. | `AGENTS.md` 파일명 규약; `src/common/naming.py:13-83`; 두 `check_completeness.py`의 `expected_score_paths`; D-15·D-16 |
| score-label 정렬 | 통과. 점수 길이는 `T_test-W-1`, 라벨은 `[W:-1]`이다. 합성 test 200·W 8에서 점수 191과 metadata `[8,-1]`이 일치했다. | GraGOD `models/predict.py:121-139`; `src/gdn_runner/run_gdn_single.py:301-313`; `docs/score_interface.md`; D-23·D-27 |
| 정규화·smoothing·집계 | 통과. 절대 오차, train·validation median-IQR, 후행 4-창, max 집계 순서다. testnorm은 부록용 별도 파일이다. GraGOD post-process는 부르지 않는다. | `configs/scoring_pipeline.yaml:4-28`; `src/gdn_runner/run_gdn_single.py:258-305`; `src/common/save_scores.py:18-56`; D-04~D-07·D-09 |
| HAI 세션 경계 | 통과. train 4세션과 test 2세션을 tuple로 유지하고 세션마다 `SlidingWindowDataset`을 만든 뒤 dataset만 합친다. test 두 세션도 각각 별도 파일명·metadata를 쓴다. | `src/data_split/load_hai_sessions.py:18-20,81-133`; `src/gdn_runner/run_gdn_single.py:111-130,160-183`; D-24·D-25 |
| resume·실패 기록 | 통과. 필수 점수·metadata·snapshot·best checkpoint·early stopping 로그를 모두 확인한 뒤에만 skip한다. HAI는 인접행렬 2벌도 요구한다. 실패 조합은 실험별 `logs/failures.csv`에 남는다. | `experiments/exp02_gdn_ghl/run_batch.py:55-108`, `check_completeness.py:49-62`; `experiments/exp03_gdn_hai_seed10/run_batch.py:60-118`, `check_completeness.py:58-73`; D-26 |
| HAI 그래프 일치도 | 통과. fit 20개마다 self-edge 포함·제거본을 저장한다. 비율별 10 seed의 45쌍, 합계 90개 Jaccard와 중앙값·사분위·범위를 낸다. | GraGOD `models/gdn/model.py:136-155`; `experiments/exp03_gdn_hai_seed10/compute_jaccard_agreement.py:18-64`; D-08·D-18·D-26 |
| 합성 단일 실행 | 통과. 봉인된 새 HEAD에서 점수 8개, metadata, 두 git hash, best checkpoint, early stopping 로그와 TopK 복원을 다시 확인했다. | `experiments/exp00_gragod_recon/dryrun_synthetic.py`; `dryrun_out/`; `tests/test_dryrun_synthetic.py`; D-27·D-31 |

## 실험 명령

아래 명령은 기록만 한다. 3b-5R3에서는 실행하지 않았다. 사용자가 실험 실행을 명시하고 GPU/CPU 환경을 정한 뒤 위에서 아래 순서로 쓴다.

```powershell
# 0. 실행 직전 봉인 확인: 첫 명령은 출력이 없어야 한다.
git status --short
git rev-parse HEAD
git -c safe.directory=C:/Users/simon/time_series/gragod-fork -C ..\gragod-fork rev-parse HEAD

# 1. GHL 시계열 01·10%·seed 1 스모크만 실행한다.
& 'C:\Users\simon\anaconda3\envs\tsad_fixed\python.exe' -c "from experiments.exp02_gdn_ghl.run_batch import run_batch; print(run_batch(specs=((1, 10, 1),)))"

# 2. 사람이 스모크 산출물을 확인한 뒤 GHL 주 배치를 실행한다.
& 'C:\Users\simon\anaconda3\envs\tsad_fixed\python.exe' experiments\exp02_gdn_ghl\run_batch.py
& 'C:\Users\simon\anaconda3\envs\tsad_fixed\python.exe' experiments\exp02_gdn_ghl\check_completeness.py

# 2-1. 주 실행 완료 뒤 뒷자르기 5%·20%를 추가 실행한다. 100%는 주 실행을 재사용한다.
& 'C:\Users\simon\anaconda3\envs\tsad_fixed\python.exe' experiments\exp02_gdn_ghl\run_batch.py --trim-direction back
& 'C:\Users\simon\anaconda3\envs\tsad_fixed\python.exe' experiments\exp02_gdn_ghl\check_completeness.py --trim-direction back

# 3. GHL 완료 뒤 HAI 주 배치와 완전성 검사를 실행한다.
& 'C:\Users\simon\anaconda3\envs\tsad_fixed\python.exe' experiments\exp03_gdn_hai_seed10\run_batch.py
& 'C:\Users\simon\anaconda3\envs\tsad_fixed\python.exe' experiments\exp03_gdn_hai_seed10\check_completeness.py

# 4. HAI 누락이 0일 때만 Jaccard 표를 만든다.
& 'C:\Users\simon\anaconda3\envs\tsad_fixed\python.exe' experiments\exp03_gdn_hai_seed10\compute_jaccard_agreement.py
```

## 3b-5 검증

이 감사 단계에서는 아래 세 명령만 한 번 실행한다. 결과를 본 뒤 이 절과 D-28을 확정한다.

```powershell
python -m unittest discover -s tests -v
& 'C:\Users\simon\anaconda3\envs\tsad_fixed\python.exe' -c "from pathlib import Path; import yaml; parsed={path.name:yaml.safe_load(path.read_text(encoding='utf-8')) for path in sorted(Path('configs').glob('*.yaml'))}; g=parsed['gdn_hyperparams.yaml']; d=parsed['data_preprocessing.yaml']; s=parsed['scoring_pipeline.yaml']; assert all(value is not None for value in parsed.values()); assert g['model_params']['topk_by_dataset']=={'GHL':5,'HAI':22}; assert g['train_params']['seeds_by_dataset']=={'GHL':[1,2,3],'HAI':list(range(1,11))}; assert d['datasets']['GHL']['ratios']==[5,10,20,50,100] and d['datasets']['HAI']['ratios']==[10,100]; assert s['error']['type']=='absolute' and s['aggregation']['mode']=='max' and s['smoothing']['window']==4; print('YAML PASS:', ', '.join(parsed))"
git diff --check
```

실행 결과는 다음과 같다.

- `python -m unittest discover -s tests -v`: 56건 통과, 실패·오류 0건
- YAML 파싱·핵심값 대조: 4개 파일 통과
- `git diff --check`: 통과

## 3b-5R1 보강 검증

- RED: runner가 `input_metadata`를 받지 못해 4건이 오류로 끝났고, 두 batch는 전달 누락으로 완료 수가 0이었다.
- GREEN: `python -m unittest tests.test_run_gdn_single tests.test_gdn_batch tests.test_dryrun_synthetic -v`에서 10건 통과, 실패·오류 0건.
- 정적 검사: 변경 Python 파일의 `compileall` 통과.
- 판정: B-02 해제. 실제 모델과 원본 데이터는 실행하지 않았다.

## 3b-5R2 보강 검증

- RED: loader 방향 인자, back 비율 제한, 통제군 완전성 함수, batch 방향 인자가 없는 상태를 각각 확인했다.
- GREEN: back split·GHL/HAI loader·GHL/HAI batch 관련 테스트 18건 통과, 실패·오류 0건.
- 정적 검사: 변경 Python 파일의 `compileall` 통과.
- 판정: B-03 해제. 실제 모델과 원본 데이터는 실행하지 않았다.

## 3b-5R3 봉인 검증

- 범위 감사: 코드·설정·문서와 exp01b·exp01c의 EDA 근거표만 commit했다. 원본 CSV, 점수 배열, checkpoint는 포함하지 않았다.
- 봉인: 3b 변경을 한 commit으로 묶었고 commit 직후 `git status --short` 출력이 비었다.
- 초기 실패: 첫 드라이런은 외부 포크의 소유자가 실행 사용자와 달라 포크 hash가 `unknown(커밋 없음)`으로 저장됐고 검사항목 d에서 멈췄다.
- 수정: `read_git_hash`가 명시된 저장소 경로만 해당 Git 호출의 `safe.directory`로 넘기도록 고쳤다. 전역 Git 설정은 바꾸지 않았다. 전용 RED→GREEN 테스트와 `tests.test_run_gdn_single` 모듈을 통과시켰다.
- 최종 실행: 고정 `tsad_fixed` 환경에서 합성 드라이런을 다시 실행했다. 검사항목 a~g가 모두 통과했다.
- 재현성: snapshot의 TSAD hash는 실행 당시 HEAD와 같고 GraGOD hash는 `485e26b0c6b1d63f4f3531c8d05597db82e9db29`다.
- 판정: B-01 해제. GHL·HAI 원본과 주 배치는 실행하지 않았다.
