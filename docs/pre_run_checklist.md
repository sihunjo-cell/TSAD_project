# GDN 4단계 진입 전 점검표

- 점검일: 2026-08-17
- 범위: 3a·3b 산출물의 1·2·3·4·5·6·7차 재감사와 보강
- 제외: 저장공간, 실제 GHL·HAI 학습, 합성 모델 학습
- 현재 판정: 3b 코드 보강은 끝났지만 새 실행본은 아직 봉인하지 않았다. 4단계는 대기한다.

## 확인 결과

| 항목 | 판정 | 근거 |
| --- | --- | --- |
| 데이터 경계 | 유지 | GHL은 파일명 `tr_`를 경계로 쓰고 HAI는 train 4세션·test 2세션 사이에 window를 만들지 않는다. `docs/manifest_draft.md`, `src/data_split/load_ghl_series.py`, `src/data_split/load_hai_sessions.py`, D-17·D-20·D-24 |
| 비율 근거 | 유지 | GHL `5·10·20·50·100%`, HAI `10·100%`, GHL back-trim `5·20·100%`를 쓴다. 5%·10%의 실행 가능성과 대표성 근거는 두 preflight 분석에 있다. `configs/data_preprocessing.yaml`, `experiments/exp01b_ghl_preflight/ANALYSIS.md`, `experiments/exp01c_hai_preflight/ANALYSIS.md`, D-21 |
| feasibility 표본 수 | 수정 | GHL·HAI 봉인 로그 665행의 정규화 표본 수를 폐기된 `L-W-1`에서 현행 `L-W`로 고쳤다. train·validation window 수와 feasibility 합격 판정은 바뀌지 않았다. 두 preflight 생성기·로그·분석, 고정 GraGOD `datasets/dataset.py:47-50,64-77`, D-34·D-41 |
| 비율·seed 원본 | 보강 | 배치와 완전성 검사는 YAML에서 조합을 읽는다. 파일명·분할·조합 코드의 허용 비율은 `SUPPORTED_RATIO_PERCENTS` 하나를 공유한다. `src/common/experiment_config.py`, `src/common/naming.py`, `src/data_split/front_trim_split.py`, D-36 |
| score-label 정렬 | 수정 | 점수 길이는 `L-W`, 라벨은 `[W:]`다. 마지막 1-step forecast를 버리지 않는다. 고정 GraGOD 포크 `datasets/dataset.py:47-50,64-77`, `models/gdn/model.py:292-316`, `src/gdn_runner/run_gdn_single.py`, `docs/score_interface.md`, D-34 |
| 전처리·라벨 | 보강 | scaler는 반환 배열을 다시 저장한다. 라벨은 길이가 맞는 유한한 0·1만 받는다. train에만 scaler를 fit하고 validation·test에는 transform만 적용한다. `src/data_split/validate_labels.py`, 두 loader, `tests/test_load_gdn_inputs.py`, D-20·D-36 |
| 실행 전 봉인 | 보강 | Git 조회 실패, dirty 작업 트리, GraGOD hash 불일치, 실행 중 HEAD·작업 트리 변경을 모두 오류로 처리한다. snapshot은 학습 전에 만든다. `src/common/verify_run_context.py`, `src/gdn_runner/run_gdn_single.py`, D-35 |
| 입력 파일 봉인 | 보강 | `configs/input_manifest.yaml`에 GHL 25개와 HAI 8개의 크기·SHA-256을 고정했다. 배치는 실행할 조합이 있을 때만 SHA-256을 한 번 검증한다. 최초 해시 전후와 실제 로드 직전·직후에는 크기·`mtime_ns`가 같아야 하며, 러너는 파일 지문이 빠진 GHL·HAI 실행을 거부한다. `src/common/verify_input_files.py`, 두 `run_batch.py`, `experiments/exp02_gdn_ghl/run_controls.py`, D-36·D-39·D-42 |
| 환경 봉인 | 보강 | Python과 package 버전을 대조하며 `pytorch-lightning`은 설치 URL과 commit `834dbf3039ee82a2ac5e65eed25f9989222283c6`도 확인한다. `configs/environment.yaml`, `src/common/verify_run_context.py`, 고정 GraGOD `requirements.txt:2`, D-36 |
| 포크·러너 계약 | 보강 | 포크의 MSE, horizon 1, Adam, ReduceLROnPlateau `factor=0.5·patience=8`, `Loss/val`, gradient clip 1.0을 실행 전에 대조한다. GraGOD `models/train.py:130-138`과 달리 우리 validation shuffle은 false로 분리한다. `configs/gdn_hyperparams.yaml`, 고정 GraGOD `datasets/dataset.py:23-37`, `models/train.py:130-138,162-192`, `gragod/training/trainer.py:119-138,206-219` |
| 후처리 고정값 | 보강 | `epsilon=0.01`, smoothing 창 4, testnorm 병행까지 실행 전에 검사한다. 창과 “처음 3점 0” 경계가 갈라지는 설정은 학습 전에 거부한다. `src/common/experiment_config.py`, `configs/scoring_pipeline.yaml`, D-04·D-07·D-15·D-39 |
| 모델 산출 유효성 | 보강 | 비유한 forecast 오차는 저장하지 않는다. HAI graph는 유한한 2차원 embedding과 0이 아닌 norm에서만 계산한다. `src/gdn_runner/run_gdn_single.py`, `src/gdn_runner/extract_adjacency.py`, 고정 GraGOD `models/gdn/model.py:139-144`, D-39 |
| 시간·완전성 | 보강 | fit마다 실제 accelerator와 학습·train-reference 추론·test 추론 시간을 `timing.json`에 나눠 저장한다. 필수 파일과 checkpoint가 0바이트가 아니어야 하며, 러너의 종료 검증과 HAI 그래프 저장까지 끝난 뒤 쓴 `COMPLETE` 표식이 있어야 완료다. 재개 전 두 저장소를 한 번 검증하고 snapshot의 두 hash가 현재 값과 같은지도 확인한다. `src/gdn_runner/run_gdn_single.py`, `src/common/run_completion.py`, 두 `check_completeness.py`, D-37·D-39~D-41 |
| GHL 대조 팔 | 보강 | −TOPK 150 fit, TopK 민감도 450 논리 조합을 고정했다. 민감도의 k=5 150개는 주 실행을 재사용하므로 대조 팔의 실제 추가 fit은 450개다. 파일과 폴더는 모델별로 분리한다. `experiments/exp02_gdn_ghl/check_controls.py`, `run_controls.py`, D-37 |
| HAI 그래프 | 보강 | best checkpoint의 embedding 유효성을 확인한 뒤 self-edge 포함·제거 인접행렬을 저장하고 제거본으로 seed 10개의 45쌍 Jaccard를 계산한다. TopK가 self를 반드시 고른다고 가정하지 않는다. Jaccard는 현재 commit의 HAI 20개 조합이 모두 끝난 뒤 YAML의 비율·seed로만 계산하며, 두 CSV 각 행에 TSAD·GraGOD commit을 적는다. `src/gdn_runner/extract_adjacency.py`, `experiments/exp03_gdn_hai_seed10/compute_jaccard_agreement.py`, D-18·D-26·D-39~D-41 |
| 근거 문서 정합성 | 수정 | Manifest 확정 전 계획서에 남은 HAI `59~86ch` 세 곳을 HAI 23.05의 확정값 `86ch`로 고쳤다. 실행 설정의 86채널, `topk=22`, PCA 30은 바뀌지 않았다. `docs/plan_v4.md`, `docs/manifest_draft.md`, D-17·D-43 |
| 재현 경로·산출물 추적 | 수정 | 패치 재검증 근거는 `experiments/exp00_gragod_recon/PATCH_REVERIFY.md`로 정확히 적는다. Jaccard의 `analysis/`는 재생성 가능한 정상 산출물이므로 `.gitignore`에 넣어 종료 clean 검사를 깨지 않게 했다. `.gitignore`, `AGENTS.md`, `configs/environment.yaml`, D-44 |
| preflight 현재성 | 수정 | GHL 분석을 다시 만들 때도 정규화 표본 `L-W`와 D-22의 W=5 확정 상태가 남는다. 오래된 “최종 W 미확정” 문구는 현재 분석과 생성기에서 함께 없앴다. `experiments/exp01b_ghl_preflight/run_ghl_preflight.py`, `ANALYSIS.md`, D-22·D-34·D-44 |
| 통계 해석 범위 | 수정 | Wilcoxon·TOST·bootstrap은 GHL 25개 task 안에서만 해석한다. 같은 simulator family라는 R1 때문에 p값·등가 판정·신뢰구간을 제조 공정 모집단으로 일반화하지 않는다. `docs/plan_v4.md:247-255`, `docs/role_C.md:34-36`, D-44 |

## 남은 봉인 절차

1. 고정 `tsad_fixed` 환경에서 전체 단위 테스트, YAML·환경 계약, compileall, `git diff --check`를 통과한다.
2. 사용자가 검토한 변경을 한 commit으로 묶고 TSAD와 GraGOD 작업 트리가 clean인지 확인한다.
3. clean HEAD에서 `experiments/exp00_gragod_recon/dryrun_synthetic.py`를 한 번 실행한다. 새 계약상 test 200·W 8이면 점수 길이 192, metadata는 `label_slice=[8,null]`이어야 한다.
4. snapshot의 TSAD hash가 실행 HEAD와 같고 GraGOD hash가 고정값인지 확인한다. 점수 8개, metadata, timing, best checkpoint, early stopping 로그, TopK 복원이 모두 있어야 한다.

이번 1·2·3·4·5·6·7차 보강은 1번까지만 수행한다. 2·3번은 사용자 검토 뒤 진행하며, 그 전에는 4단계 GHL 스모크를 실행하지 않는다. D-27의 191점 합성 결과는 과거 구현 기록일 뿐 새 봉인 근거가 아니다.

## 1차 보강 검증값

- 고정 `tsad_fixed` 환경 전체 단위 테스트: 91건 통과, 실패·오류 0건
- runtime: Python 3.10.20과 package 11개 버전 일치
- `pytorch-lightning` 설치 원본: `https://github.com/gonzachiar/pytorch-lightning.git@834dbf3039ee82a2ac5e65eed25f9989222283c6` 일치
- YAML: 5개 파싱 통과, manifest GHL 25개·HAI 8개 확인
- 조합: GHL 주 실행 375, back-trim 225, −TOPK 150, TopK 민감도 450, HAI 20 확인
- `python -m compileall -q src experiments tests`: 종료 코드 0
- `git diff --check`: 종료 코드 0
- GraGOD 포크: HEAD `485e26b0c6b1d63f4f3531c8d05597db82e9db29`, 작업 트리 clean

모델 학습과 전체 CSV 재검사는 실행하지 않았다.

## 2차 보강 검증값

- 봉인 로그 재계산: GHL 25개·19채널·학습 길이 39,938~50,000·범위 이탈 0, feasibility 실패 0. HAI train 4세션·896,400행·86채널·feasibility 실패 0
- GHL 고정 채널 재대조: 5% `unique=1` 154/475, IQR=0 279/475. 100%는 각각 0/475, 150/475
- HAI 활성 채널 재대조: 10% 59개, 100% 66개. `topk=ceil(0.25×86)=22`
- 고정 `tsad_fixed` 환경 전체 단위 테스트: 96건 통과, 실패·오류 0건
- YAML 5개 파싱, 조합 수 `375·225·150·450·20`, compileall, `git diff --check` 통과
- GraGOD 포크: HEAD `485e26b0c6b1d63f4f3531c8d05597db82e9db29`, 작업 트리 clean

2차 감사에서도 모델 학습과 전체 CSV·SHA-256 재검사는 실행하지 않았다.

## 3차 보강 검증값

- 3a 재대조: GHL 25개·19채널·학습 길이 39,938~50,000, HAI 86채널·훈련 4세션·테스트 2세션 유지
- 비율·조합 재대조: GHL `5·10·20·50·100%`, HAI `10·100%`, `topk=5·22`, 조합 수 `375·225·150·450·20` 일치
- 고정 `tsad_fixed` 환경 전체 단위 테스트: 99건 통과, 실패·오류 0건
- 이전 commit snapshot과 형식이 깨진 snapshot을 완료로 세지 않고, 미완료 HAI graph를 Jaccard에 섞지 않는 회귀 테스트 통과
- 고정 Lightning `prediction_loop.py:273-274`의 CPU 이동 확인. CUDA 장치 불일치 의혹은 기각
- YAML 5개와 Manifest GHL 25개·HAI 8개, 조합 수 `375·225·150·450·20` 재확인
- Python 3.10.20, package 11개, 고정 Lightning 설치 원본 확인
- `compileall`과 `git diff --check` 종료 코드 0
- GraGOD 포크: HEAD `485e26b0c6b1d63f4f3531c8d05597db82e9db29`, 작업 트리 clean

3차 감사도 모델 학습, 전체 CSV·SHA-256 재검사, 새 commit을 수행하지 않았다.

## 4차 보강 검증값

- 실제 kept length 23종에서 validation 10%의 float ceil과 정확 유리수 ceil이 모두 일치
- GHL 625행·HAI 40행의 kept, train, validation, window, 정규화 표본 수 수식 대조 통과
- 설정 불변식: `min_train_length=W+1`, `topk={5,22}`, 대조 팔 부분집합과 조합 수 `375·225·150·450·20` 일치
- 새 회귀 테스트에서 `timing.json.accelerator`와 두 Jaccard CSV의 TSAD·GraGOD commit 확인
- 고정 `tsad_fixed` 환경 전체 단위 테스트: 100건 통과, 실패·오류 0건
- YAML 5개, 환경 12항목, compileall, `git diff --check` 통과

4차 감사도 모델 학습, 합성 드라이런, 원본 CSV·SHA-256 재검사, 새 commit을 수행하지 않았다.

## 5차 보강 검증값

- D-34~D-41에서 점수 길이 교정은 한 번뿐이며 비율·seed·validation·window·topk의 번복은 없음을 확인
- manifest SHA-256 검증 뒤 로드 중 입력을 바꾸는 실패 테스트가 GHL 주 배치·대조군·HAI에서 모델 호출 전에 모두 차단됨
- 시작 시 SHA-256 1회, 이후 로드 직전·직후 크기·`mtime_ns` 확인으로 장시간 배치의 일반적인 입력 변경을 감지
- 고정 `tsad_fixed` 환경 전체 단위 테스트: 104건 통과, 실패·오류 0건
- YAML 5개, 환경 12항목, 조합 수 `375·225·150·450·20`, compileall, `git diff --check` 통과
- GraGOD 포크: HEAD `485e26b0c6b1d63f4f3531c8d05597db82e9db29`, 작업 트리 clean

5차 감사도 모델 학습, 합성 드라이런, 원본 CSV·SHA-256 재검사, 새 commit을 수행하지 않았다. 새 diff나 실패 증거가 없으면 3단계 정적 감사를 다시 열지 않는다.

## 6차 감사 검증값

- D-34~D-42의 확정값과 설정→loader→GraGOD 학습·예측→점수→snapshot→완료·재개 경로를 다시 대조
- 비율·seed·validation·window·topk, `L-W` 정렬, 조합 수 `375·225·150·450·20` 유지
- 실행 코드의 새 결함은 재현되지 않아 코드·테스트를 바꾸지 않음
- `docs/plan_v4.md`의 HAI 채널 범위 세 곳만 `86ch`로 정정
- 수정 전 고정 `tsad_fixed` 환경 전체 단위 테스트: 104건 통과, 실패·오류 0건

6차 감사도 모델 학습, 합성 드라이런, 원본 CSV·SHA-256 재검사, 새 commit을 수행하지 않았다. 문서 수정 뒤에는 검증 비용 규율에 따라 경로·수치·diff만 확인한다.

## 7차 최종 감사 검증값

- 설정→loader→GraGOD 학습·예측→점수→snapshot→완료·재개 경로에서 실행 알고리즘의 새 결함은 재현되지 않음
- 비율·seed·validation·window·topk, `L-W` 정렬, 조합 수 `375·225·150·450·20` 유지
- `experiments/exp00_gragod_recon/PATCH_REVERIFY.md` 실재와 고정 포크 소스 줄 재확인
- Jaccard 산출물은 수정 전 `NOT_IGNORED`, 수정 뒤 `.gitignore`의 `experiments/exp03_gdn_hai_seed10/analysis/` 규칙으로 제외됨
- GHL preflight 관련 단위 테스트 7건 통과, 실패·오류 0건
- 전체 테스트 104건은 5차와 6차에 통과했고 이번에는 실행 논리를 바꾸지 않아 다시 돌리지 않음

7차 감사도 모델 학습, 합성 드라이런, 원본 CSV·SHA-256 재검사, 새 commit을 수행하지 않았다.

## 다음 실행 명령

```powershell
& 'C:\Users\simon\anaconda3\envs\tsad_fixed\python.exe' -m unittest discover -s tests -v
& 'C:\Users\simon\anaconda3\envs\tsad_fixed\python.exe' -m compileall -q src experiments tests
git diff --check
```

실제 모델 실행은 위 정적 검증, 사용자 검토, commit 봉인이 끝난 뒤 별도 단계에서 한다.
