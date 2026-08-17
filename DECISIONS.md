# 결정 로그 — 레포 확인·규약 변경은 전부 여기에 한 줄씩

- 2026-08-14 exp00: GraGOD 레포는 github.com/GraGODs/GraGOD가 맞다. README.md:112가 d-ailin/GDN 기반임을 밝히고 있어서다. develop HEAD는 ec8cd452a410ba903a31beb097a010ba0448c095였고, 고정 해시 선택은 이때 보류했다(후보 3개는 RECON.md [A]).
- 2026-08-14 exp00: GDN 논문 값은 arXiv 2106.06947의 ar5iv HTML 렌더(§4.4, §3.6)에서 확인했다. PDF를 직접 파싱할 수 없는 환경이었다.
- 2026-08-14 exp00: 커밋이 "GDN 관련"인지는 두 기준으로 판정한다. models/gdn/** 을 직접 바꿨으면 ●, GDN이 함께 쓰는 datasets/·gragod/predictions/·gragod/training/·models/predict.py를 바꿨으면 ○.
- 2026-08-14 규약 변경: 코드 스타일에 ponytail 원칙을 추가했다(간결·직관·최소, 추측성 추상화 금지). 사용자가 직접 지시했다.
- 2026-08-14 규약 변경: 문서 작성 스타일 절을 새로 만들었다. 한국어 산문은 humanize-korean 룰북(github.com/epoko77-ai/im-not-ai, 로컬 클론 ../im-not-ai @53e24e8)을 쓰는 시점부터 적용한다. 이것도 사용자 직접 지시다.
- 2026-08-14 윤문 승인: 동결돼 있던 plan_v4.md·role_A/B/C.md·DECISIONS.md 결정 문구를 "의미 불변 + 문체만" 조건으로 humanize-korean 윤문했다. 사용자가 직접 승인했다(D-19의 재구성 금지·부탁문 원문 유지 지시에 대한 1회 예외이며, 수치·해시·절 번호·참조는 그대로다).
- 2026-08-16 exp01b: GHL 파일명의 `tr_`는 앞쪽 학습 행 수, `1st_`는 전체 파일의 0-based 첫 이상 인덱스로 읽는다. 공식 목록 `TheDatumOrg/TSB-AD/Datasets/File_List/TSB-AD-M.csv:33-57`과 실데이터 25개를 대조해 모두 일치했다(`experiments/exp01b_ghl_preflight/ANALYSIS.md:18-22`).
- 2026-08-16 exp01b, 2026-08-17 집계 문구 정정: 비율별 앞쪽 학습 구간에서 median·IQR을 다시 추정하고 IQR=0 채널도 임의로 버리지 않는다. 5% 구간에서 `unique_value_count=1`인 완전 고정 행은 154/475개지만 학습 전체에서는 0/475개다. IQR=0은 279/475개다(`experiments/exp01b_ghl_preflight/logs/train_channel_quality.csv`; `ANALYSIS.md:23-25`).
- 2026-08-16 exp01b: 5·10·20·50·100% 격자는 잠정 유지한다. 학습 구간 EDA에서 5%는 스트레스 하한, 10%는 저데이터 기준점, 20%는 구조 회복 지점, 50%는 고데이터 대조, 100%는 전체 기준으로 판정했다(`experiments/exp01b_ghl_preflight/ANALYSIS.md:28-46`). 강혁 산출물이 데이터 계약이나 안정화 전처리 근거를 바꾸면 같은 검사를 다시 돌려 수정한다(`experiments/exp01b_ghl_preflight/ANALYSIS.md:52`).

## 확정 결정 (2026-08-14, 결정 기록 세션)

### D-01
- 결정: GraGOD는 ec8cd452 커밋을 베이스로 잡고, 여기에 D-03 패치를 얹은 우리 포크의 커밋을 최종 고정 대상으로 삼는다. 패치 커밋 해시는 패치 세션을 마친 뒤 이 항목에 덧붙인다.
- 근거: RECON [A]. SHAPEFLOW 검증도 이 커밋 위에서 수행됐다.
- 기록일: 2026-08-14
- 패치 커밋 해시: 485e26b0c6b1d63f4f3531c8d05597db82e9db29 — 브랜치 fix/gdn-input-transform, 베이스 ec8cd452. 2026-08-14 패치 세션에서 기입했고, `experiments/exp00_gragod_recon/PATCH_REVERIFY.md`의 재검증을 전 항목 통과했다.

### D-02
- 결정: 데이터 주입은 RECON [B]의 후보 (ii)로 간다. GraGOD의 get_data_loader에 우리가 앞자르기한 텐서를 직접 넣는 자체 러너를 만들고, GraGOD의 train.py/predict.py 오케스트레이션은 쓰지 않는다.
- 근거: RECON [B]. SHAPEFLOW 2부가 이미 이 방식으로 실행됐다.
- 기록일: 2026-08-14

### D-03
- 결정: models/gdn/model.py 271행과 304행의 reshape(-1, x.size(2), x.size(1))을 permute(0, 2, 1).contiguous()로 바꾼다. 117행의 view(-1, all_feature)는 원본 d-ailin과 공유하는 줄이라 건드리지 않는다. 내부 수정 최소화 원칙의 유일한 예외이며, 패치 후 우리가 쓰는 것은 GraGOD 원형이 아니라 패치 포크라는 사실을 명시한다.
- 근거: VERIFICATION 의혹 1이 참으로 판정됐다. SHAPEFLOW 종합 판정은 두 줄만으로는 불충분(e1에서 117행 view가 연속성을 요구), e2에서 contiguous를 붙이면 전 경로가 정합함을 실증했고, 1부 d의 원본 대조로 포팅 의도가 (batch, n_features, window) 입력임을 확정했다.
- 기록일: 2026-08-14

### D-04
- 결정: raw와 smoothed를 이렇게 정의한다. raw는 채널별 정규화 점수를 max로 집계한 배열이다. smoothed는 채널별 정규화 점수에 시간축 후행 4-창 smoothing(d-ailin evaluate.py:62-65 방식, 처음 3개 시점은 0)을 채널별로 적용한 뒤 max로 집계한 배열이다. 순서는 정규화 → (smoothing) → 집계로 고정한다.
- 근거: RECON [C]·[G]
- 기록일: 2026-08-14

### D-05
- 결정: GraGOD의 smooth_scores는 어떤 경로에서도 부르지 않는다 — feature 축에 작용한다는 것이 VERIFICATION 의혹 2에서 참으로 확인됐기 때문이다. d-ailin의 후행 4-창을 우리 코드로 직접 구현하고, VERIFICATION의 d-ailin 재현 출력과 단위 테스트로 대조한다. 1단계에서 걸어 뒀던 smoothing 승인 보류는 이 결정으로 푼다.
- 근거: VERIFICATION 의혹 2 = 참 (대조 표 포함)
- 기록일: 2026-08-14

### D-06
- 결정: 정규화는 post_process_scores=false로 정규화 전 절대 오차를 받아, 학습/검증 구간의 채널별 median·IQR을 우리 코드로 추정해 테스트 점수에 적용한다. 저자 원본 방식(테스트셋 추정)으로 계산한 결과도 부록 대조용으로 함께 만들되 파일명으로 구분한다.
- 근거: RECON [C] 방법 a, RECON [D], 계획서 7-3
- 기록일: 2026-08-14

### D-07
- 결정: IQR epsilon은 d-ailin의 1e-2를 쓴다. 값은 코드가 아니라 configs가 소유한다.
- 근거: RECON [D] — GraGOD는 epsilon이 없어 0으로 나눌 위험이 있다.
- 기록일: 2026-08-14

### D-08
- 결정: 인접행렬 추출은 RECON [F]의 후보 2로 간다. best.ckpt의 embedding.weight로 cos-sim → topk를 다시 계산한 "best validation 시점의 그래프"를 exp03 산출물로 삼고, 재계산 로직의 출처(model.py:137-144)를 주석으로 남긴다. Jaccard를 계산하기 전에 실제 self-edge를 제거한다. self-edge는 채널 관계가 아니며 `GraphLayer.forward`가 TopK 입력에서 제거한 뒤 모든 노드에 다시 붙이므로 시드 간 관계 일치도에 넣지 않는다.
- 근거: RECON [F]. SHAPEFLOW 1부 a에서 그래프가 embedding에만 의존한다는 것을 확인했으므로 재계산은 결정적이다.
- 기록일: 2026-08-14

### D-09
- 결정: 채널 집계는 max로 고정한다(계획서 7-5). GraGOD swat 설정의 mean과 다르다는 점을 명기한다.
- 근거: RECON [G]
- 기록일: 2026-08-14

### D-10
- 결정: torch-geometric은 GraGOD 어디에도 선언돼 있지 않으므로, torch 2.2.2와 호환되는 버전을 우리가 골라 configs/environment에 정확히 고정한다. 버전 선정은 패치 세션에서 설치 검증과 함께 한다.
- 근거: RECON [H]. SHAPEFLOW는 스크래치 2.8.0.post1로 수행됐고 고정 환경 판정이 아니라는 것을 스스로 밝혀 뒀다.
- 기록일: 2026-08-14
- 확정 버전(2026-08-14 패치 세션): torch-geometric==2.5.3. 전체 고정 환경은 configs/environment.yaml에 있다 — python 3.10.20, torch 2.2.2+cpu, numpy 1.26.4, pytorch-lightning은 저자 포크 커밋 834dbf30(설치 성공, 대체 없음), tensorboardX 2.6.2.2. 선정 근거는 `experiments/exp00_gragod_recon/PATCH_REVERIFY.md` [1].

### D-11
- 결정: GraGOD에 하드코딩된 시드 42는 쓰지 않는다. 우리 러너가 set_seeds를 우리 시드로 직접 부른다.
- 근거: RECON [B]와 보류 목록 11번. SHAPEFLOW 2부가 이미 러너 방식으로 실행한 전례가 있다.
- 기록일: 2026-08-14

### D-12
- 결정: 패치 세션에서 D-03을 적용한 뒤 다음 세 가지를 고정 환경에서 실행으로 다시 확인해야 3b가 열린다. 첫째, SHAPEFLOW e2와 같은 장난감 입력에서 forward가 돌고 출력 shape가 (batch, n_features)여야 한다. 둘째, SHAPEFLOW f와 같은 설계에서 채널 오프셋이 해당 채널 자리에만 나타나야 한다. 셋째, SHAPEFLOW g와 같은 설계에서 y 정렬이 유지돼야 한다. 기대 출력의 원본은 SHAPEFLOW 2부의 실행 출력이다.
- 근거: SHAPEFLOW 2부 (e2·f·g 실행 출력)
- 기록일: 2026-08-14

### D-13
- 결정: VERIFICATION과 SHAPEFLOW는 torch 2.10 스크래치 환경에서 수행됐다는 점을 기억해 둔다. D-12의 재검증은 고정 환경(torch 2.2.2 + D-10 확정 버전)에서 한다.
- 근거: VERIFICATION·SHAPEFLOW의 환경 고지 절
- 기록일: 2026-08-14

### D-14
- 결정: (질문1 해소) 포크는 GitHub fork도 vendored 복사본도 아니고, 우리 저장소 바깥의 로컬 저장소로 둔다. 위치는 우리 저장소와 같은 부모 폴더 아래 gragod-fork/ 이고, 이 경로를 CLAUDE.md에 적는다. 재현 경로는 세 가지 — 베이스 ec8cd452, patches/gdn_input_transform.diff, 패치 커밋 해시 — 로 우리 저장소 안에서 닫힌다.
- 근거: 결정 기록 세션(2026-08-14)의 질문 목록 1번; D-01·D-03
- 기록일: 2026-08-14

### D-15
- 결정: (질문2 해소) D-04의 "정규화 점수"는 D-06의 학습/검증 구간 추정 정규화를 가리킨다. 저자 원본 방식(테스트셋 추정)의 부록 산출물은 파일명 접미사로 구분한다. 파일명 규약을 {dataset}__{series}__{model}__{tier}__r{ratio}__s{seed}__{raw|smoothed}__{trainnorm|testnorm}.npy 로 넓히고, 기본 산출물은 trainnorm, 부록 대조는 testnorm으로 한다.
- 근거: 결정 기록 세션(2026-08-14)의 질문 목록 2번; D-04·D-06
- 기록일: 2026-08-14

### D-16
- 결정: (질문3 해소) 집계 전 채널별 점수 배열은 접미사 __channels를 붙인 보조 산출물로 보존한다. 1급 산출물은 집계 후 배열이라는 격 구분은 그대로 둔다.
- 근거: 회수 분석에서 신호원 채널을 추적할 수 있고, 모델을 다시 돌리지 않고 재집계할 수 있어서다(계획서 5-4의 재채점 비용 0 원칙).
- 기록일: 2026-08-14

### D-17
- 결정: HAI는 23.05의 86채널 데이터를 쓰고, CSV 한 파일을 연속 시계열 하나로 센다. 훈련 4개와 테스트 2개 사이에는 윈도를 걸치지 않는다. N=86이므로 k=max(5, ceil(0.25N))=22로 확정한다.
- 근거: `docs/manifest_draft.md`; icsdataset/hai `README.md:24, 34, 58-95, 240-246`; `hai_dataset_technical_details.pdf` PDF 2·31쪽; 계획서 7-4
- 기록일: 2026-08-16

### D-18
- 결정: (질문6 해소) Jaccard 비교는 시드 쌍별 전수 비교다. seed 10개가 만드는 45쌍 전부의 Jaccard(self-edge 제거 후 edge 집합 기준, D-08)를 계산해 중앙값·사분위·범위의 분포로 보고한다. 기준 그래프 하나를 정해 대조하는 방식은 기준 선택이 자의적이라 쓰지 않는다.
- 근거: 결정 기록 세션(2026-08-14)의 질문 목록 6번; D-08
- 기록일: 2026-08-14

### D-19
- 결정: (질문4 해소) 계획서 참조가 저장소 안에서 닫히도록 docs/plan_v4.md 를 근거 문서 위치로 지정한다. 이 파일의 내용은 사용자가 확정본을 직접 넣는다 — 세션이 계획서를 재구성하거나 요약해 채우지 않는다. 그 세션에서는 빈 파일과 "사용자가 확정본을 붙여넣을 자리"라는 한 줄 안내만 만들었다.
- 근거: 결정 기록 세션(2026-08-14)의 질문 목록 4번
- 기록일: 2026-08-14

### D-20
- 결정: GHL·HAI는 다운샘플 배율 1, 초기 절단 0포인트로 고정하고 timestamp·label을 모델 입력에서 제외한다. validation 분할 뒤 train 부분에만 `MinMaxScaler`를 fit하며 같은 scaler로 validation·test를 변환한다. GHL은 현재 시계열의 train 부분마다 scaler 하나를 fit한다. HAI는 세션별로 비율과 validation을 자른 뒤 네 train 부분으로 scaler 하나를 fit하고 파일 경계에는 윈도를 만들지 않는다.
- 근거: `configs/data_preprocessing.yaml`; `experiments/exp01c_hai_preflight/ANALYSIS.md`; icsdataset/hai `README.md:229-264, 284-294`; HAI 기술문서 PDF 2쪽; GraGOD `datasets/config.py:54-90`, `datasets/data_processing.py:54-70`, `datasets/swat.py:119-149`; 계획서 4절·7-1~7-3
- 기록일: 2026-08-16

### D-21
- 결정: HAI GDN은 계획서대로 10%와 100%를 쓴다. 10%는 각 훈련 세션 앞 10% 네 조각이며 충분 데이터가 아니라 저데이터 대표 조건으로 해석한다. 네 조각은 12,600~29,160행이고, 전체 훈련에서 움직이는 66채널 중 59채널을 관측한다. W 후보 5·15·50·55·155 전부에서 train·validation·정규화 표본 부족 조건은 0개다.
- 근거: `experiments/exp01c_hai_preflight/logs/ratio_feasibility.csv`, `ratio_representativeness.csv`, `corpus_channel_activity.csv`; `experiments/exp01c_hai_preflight/ANALYSIS.md`; 계획서 4절
- 기록일: 2026-08-16

### D-22
- 결정: GDN 주 설정은 window 5, embedding 64, output layer 1개·hidden 128, heads 1, dropout 0.2, negative slope 0.2, learned graph 사용으로 고정한다. 학습은 batch 32, 최대 50 epoch, Adam `lr=0.001`, `weight_decay=0`, `eps=1e-8`, `betas=(0.9,0.99)`, early stopping `patience=10`, `min_delta=0`, validation 0.1을 쓴다. GHL·HAI topk는 5·22, seed는 각각 1~3·1~10이다. 모든 비율에 같은 값을 쓰며 결과를 본 뒤 조정하지 않는다.
- 근거: `docs/gdn_hyperparameter_decisions.md`; d-ailin/GDN `run.sh:3-18`, `main.py:183-197`, `train.py:27,44,86-95`, `models/GDN.py:76-105,138-150`; GDN 논문 §4.4; GraGOD `models/gdn/params.yaml:1-32`, `models/gdn/model.py:44-56,136-159`, `gragod/training/trainer.py:119-138,206-228`; D-17
- 기록일: 2026-08-16

### D-23
- 상태: 점수 길이와 라벨 범위는 D-34가 대체한다. 나머지 러너 책임은 유지한다.
- 결정: GDN 단일 러너는 비율 적용과 전처리가 끝난 train·test 배열만 받는다. config의 `val_size`만큼 train 뒤쪽을 validation으로 떼고, 축소된 train 전체의 예측 오차로 trainnorm 통계를 추정한다. GraGOD 의존성은 `load_gdn_dependencies` 한 곳에서 불러온다. 모델 생성 시 `edge_index`, `n_features`, `out_dim`을 주입하며, 점수 길이는 `T_test-W-1`, label 범위는 `[W:-1]`로 기록한다. 반환값에는 8개 점수 경로, metadata, best checkpoint, early stopping 로그, config·두 저장소 git hash snapshot 경로를 넣는다.
- 근거: `tests/test_run_gdn_single.py`; `src/gdn_runner/run_gdn_single.py`; GraGOD `models/train.py:115-204`, `models/predict.py:121-139,363-366`, `gragod/training/callbacks.py:11-59`, `gragod/training/trainer.py:163-228`; `experiments/exp00_gragod_recon/ORCHESTRATION.md`
- 기록일: 2026-08-16

### D-24
- 결정: GHL·HAI 입력 로더는 train·validation·test를 세션별 tuple로 반환한다. GHL은 파일명 `tr_` 경계와 5·10·20·50·100%를 쓰고 현재 시계열 train 부분에 scaler를 fit한다. HAI는 10·100%만 허용하며 네 세션을 각각 앞자르기·validation한 뒤 네 train 부분에 MinMaxScaler 하나를 `partial_fit`한다. timestamp·label은 feature에서 빼고 test·label timestamp가 같은지 확인한다. HAI 세션은 raw tensor로 연결하지 않는다.
- 근거: `src/data_split/load_ghl_series.py`; `src/data_split/load_hai_sessions.py`; `tests/test_load_gdn_inputs.py`; `configs/data_preprocessing.yaml`; `docs/manifest_draft.md`; GraGOD `datasets/dataset.py:64-77,97-143`, `datasets/data_processing.py:54-70`; D-17·D-20·D-21
- 기록일: 2026-08-16

### D-25
- 결정: 다중 세션 러너는 세션마다 `SlidingWindowDataset`을 만들고 dataset만 `ConcatDataset`으로 합친다. trainnorm 통계는 같은 원 세션의 train·validation을 복원해 오차를 구한 뒤 세션별 오차 배열을 합쳐 추정한다. 모델 학습과 best checkpoint 로드는 실행당 한 번이며 HAI test 2세션은 같은 모델로 각각 점수 8개와 metadata를 저장한다. 기존 `run_gdn_single`은 validation 분할 뒤 한 세션 tuple로 다중 세션 본체를 호출한다.
- 근거: `src/gdn_runner/run_gdn_single.py`; `tests/test_run_gdn_single.py`; GraGOD `datasets/dataset.py:7-77,97-143`, `models/train.py:119-145`, `models/predict.py:121-139,363-366`; D-23·D-24
- 기록일: 2026-08-16

### D-26
- 결정: exp02는 `25×5×3=375`, exp03은 `2×10=20`개 학습 조합을 고정한다. 조합마다 별도 `runs/` 하위 폴더를 써서 체크포인트·스냅숏·로그가 덮이지 않게 한다. 필수 점수·metadata·best checkpoint·early stopping 로그·config 스냅숏이 모두 있을 때만 완료로 보고 다시 실행하지 않는다. 실패는 각 실험의 `logs/failures.csv`에 남긴다. HAI는 best checkpoint에서 TopK edge를 재계산해 self-edge 포함본과 제거본을 모두 저장한다. Jaccard는 제거본으로 비율마다 seed 10개의 45쌍을 전수 비교하고 중앙값·사분위·범위를 낸다.
- 근거: `experiments/exp02_gdn_ghl/run_batch.py`, `check_completeness.py`; `experiments/exp03_gdn_hai_seed10/run_batch.py`, `check_completeness.py`, `compute_jaccard_agreement.py`; `tests/test_gdn_batch.py`; GraGOD `models/gdn/model.py:136-155`, `gragod/training/trainer.py:18-41`; D-08·D-18·D-22·D-25
- 기록일: 2026-08-16

### D-27
- 상태: 이 실행은 당시 구현을 기록한 과거 검증이다. 점수 정렬 판정은 D-34가 대체하며 새 구현의 봉인 드라이런으로 쓰지 않는다.
- 결정: 3b-4 합성 드라이런은 `tsad_fixed`의 Python 3.10.20에서 train `(300, 5)`, test `(200, 5)`, window 8, 2 epoch로 한 번 실행했다. 집계 배열 4개는 `(191,)`, 채널별 배열 4개는 `(191, 5)`였고 전 값이 유한했다. `191=200-8-1`이며 metadata의 `test_length=200`, `score_length=191`, `label_slice=[8,-1]`과 일치한다. snapshot에는 현재 저장소 `6876b3de9d6968cd1f20dcf8c3cd3817c440227e`와 고정 포크 `485e26b0c6b1d63f4f3531c8d05597db82e9db29`가 기록됐다. best checkpoint와 early stopping 로그가 남았고, checkpoint에서 TopK edge 10개와 self-edge 제거본 5개를 복원했다. 검사항목이 하나라도 틀리면 드라이런을 실패시키도록 판정 함수도 고쳤다. 실제 Lightning checkpoint를 별도 프로세스에서 열 때는 고정 포크 경로를 먼저 import 경로에 넣어야 한다. 드라이런과 HAI 배치는 `run_gdn_sessions`가 같은 프로세스에서 포크를 먼저 등록하므로 이 조건을 충족한다.
- 근거: `experiments/exp00_gragod_recon/dryrun_synthetic.py`; `tests/test_dryrun_synthetic.py`; `experiments/exp00_gragod_recon/dryrun_out/`; GraGOD `models/predict.py:121-139,363-366`, `models/gdn/model.py:313-316`, `gragod/training/callbacks.py:35-50`; D-08·D-23·D-25
- 기록일: 2026-08-16

### D-28
- 결정: 3b-5 감사 결과, 주 GDN 설정·데이터 경계·배치·점수 계약은 맞지만 실제 실험 착수는 보류한다. 차단점은 세 가지다. 현재 TSAD 작업 트리가 dirty라 snapshot commit이 실행 코드를 복원하지 못하고, 실행 snapshot에 전처리·점수 설정 전문과 입력 출처·세션 분할이 빠져 있으며, 계획서 7-1의 GHL `{100,20,5%}` 뒷자르기 통제군은 분할 함수만 있고 실행 경로가 없다. 수정은 snapshot 계약, 뒷자르기 배치, clean commit 봉인 순서로 나눈다. 로컬 고정 환경이 `torch 2.2.2+cpu`라는 점도 전체 배치 실행 환경 선택 때 별도로 확인한다.
- 근거: `docs/pre_run_checklist.md`; `git status --short`; `experiments/exp00_gragod_recon/dryrun_out/snapshots/config_snapshot.json`; `src/gdn_runner/run_gdn_single.py:270,322-328`; `src/common/save_scores.py:83-98`; `docs/plan_v4.md:213`; `src/data_split/back_trim_split.py`; `experiments/exp02_gdn_ghl/run_batch.py`; 전체 단위 테스트 56건 및 YAML 4개 대조 통과
- 기록일: 2026-08-16

### D-29
- 결정: 실행 snapshot의 `config`에 `run_config`, `data_preprocessing`, `scoring_pipeline`, `input`, seed, 두 저장소 git hash를 함께 저장한다. `input`은 batch가 받은 데이터 경로와 로더의 feature 이름·`session_splits`를 그대로 담는다. 설정 복사본을 따로 만들거나 값을 다시 계산하지 않는다. 이 계약으로 B-02를 해제하며 B-01·B-03은 유지한다.
- 근거: `src/gdn_runner/run_gdn_single.py:275-278,329-338`; `experiments/exp02_gdn_ghl/run_batch.py:94-106`; `experiments/exp03_gdn_hai_seed10/run_batch.py:100-112`; `tests/test_run_gdn_single.py:228-280,301-368`; `tests/test_gdn_batch.py:91-180`; GraGOD `models/train.py:115-204`; 관련 테스트 10건과 변경 파일 compile 통과
- 기록일: 2026-08-17

### D-30
- 결정: GHL 뒷자르기 통제군은 5%·20%·100%의 225개 논리 조합으로 관리한다. 5%·20%의 150개 추가 fit은 exp02의 `back_trim_runs/`에 저장하고 실패는 `back_trim_logs/`에 남긴다. 100%는 앞·뒤 splitter와 loader 출력이 모두 같으므로 주 실행 `runs/`에 한 번만 저장한다. 통제군 완전성 검사는 이 75개 주 실행 산출물을 같은 비교점으로 읽는다. snapshot의 `run_config.trim_direction`과 원본 인덱스 범위가 분할 방향을 기록한다.
- 근거: `docs/plan_v4.md:213`; `src/data_split/front_trim_split.py:34-45`; `src/data_split/back_trim_split.py:14-25`; `src/data_split/load_ghl_series.py:20-93`; `experiments/exp02_gdn_ghl/check_completeness.py:14-106`; `experiments/exp02_gdn_ghl/run_batch.py:48-151`; `tests/test_back_trim_split.py:30-40`; `tests/test_load_gdn_inputs.py:89-149`; `tests/test_gdn_batch.py:142-250`; GraGOD `datasets/data_processing.py:54-70`, `models/train.py:119-145`; 관련 테스트 18건과 변경 파일 compile 통과
- 기록일: 2026-08-17

### D-31
- 결정: 3b 준비 변경은 원본 데이터와 모델 산출물을 제외하고 한 commit으로 봉인한다. commit 직후 clean 작업 트리에서 `tsad_fixed` 합성 드라이런을 한 번 실행하며, snapshot의 TSAD hash가 실행 당시 HEAD와 같고 GraGOD hash가 고정값 `485e26b0c6b1d63f4f3531c8d05597db82e9db29`일 때만 B-01을 해제한다. 점수 8개와 metadata, best checkpoint, early stopping 로그, TopK 복원이 모두 있어야 한다. TSAD hash는 문서에 복제하지 않고 `dryrun_out/snapshots/config_snapshot.json`을 원본으로 삼아 commit을 다시 만들 때 생기는 자기 참조를 피한다. 이 조건을 충족해 3b 준비를 닫았으며 GHL·HAI 원본과 주 배치는 실행하지 않았다.
- 근거: `git status --short`; `experiments/exp00_gragod_recon/dryrun_synthetic.py:44-127`; `experiments/exp00_gragod_recon/dryrun_out/snapshots/config_snapshot.json`; `src/gdn_runner/run_gdn_single.py:293-338`; GraGOD `models/predict.py:121-139,363-366`, `models/gdn/model.py:313-316`, `gragod/training/callbacks.py:35-50`; `docs/pre_run_checklist.md` 3b-5R3 봉인 검증
- 기록일: 2026-08-17

### D-32
- 결정: 실행 snapshot에서 외부 저장소 hash를 읽을 때는 호출자가 명시한 저장소의 절대 경로만 `git -c safe.directory=<경로>`에 넘긴다. sandbox·container처럼 실행 사용자와 포크 소유자가 다른 환경에서도 hash를 기록하되 전역 Git 설정은 바꾸지 않는다. 첫 봉인 드라이런에서 포크 hash가 `unknown(커밋 없음)`으로 기록된 실제 실패를 재현한 뒤 이 범위만 고쳤다.
- 근거: `src/gdn_runner/run_gdn_single.py:76-85`; `tests/test_run_gdn_single.py`의 `TestReadGitHash`; 첫 실패 snapshot과 Git stderr의 `detected dubious ownership`; 최종 `experiments/exp00_gragod_recon/dryrun_out/snapshots/config_snapshot.json`
- 기록일: 2026-08-17

### D-33
- 결정: 4단계 전 1차 재감사는 3b 구현 전체를 다시 열어 데이터 경계, 점수 정렬, 실행 봉인, 설정 원본, 대조 팔을 한 흐름으로 점검한다. 현재 변경은 새 commit과 clean 상태의 합성 드라이런 전까지 봉인된 실행본으로 보지 않는다. D-31의 과거 봉인은 그 당시 commit에만 유효하다.
- 근거: `docs/superpowers/specs/2026-08-17-stage3-hardening-design.md`; `docs/superpowers/plans/2026-08-17-stage3-hardening.md`; `docs/pre_run_checklist.md`
- 기록일: 2026-08-17

### D-34
- 결정: GDN 1-step forecast의 점수 길이는 `L-W`, 라벨 범위는 `[W:]`로 고친다. `SlidingWindowDataset`은 길이 L에서 `L-W`개 target을 만들며 마지막 target은 `X[L-1]`이다. GraGOD GDN의 `post_process_predictions`와 공통 predict 경로가 마지막 값을 버리는 동작은 reconstruction 설명에서 온 것이므로 쓰지 않는다. 러너가 predict 출력 전체를 이어 붙여 `X[W:]`와 직접 절대 오차를 계산한다.
- 근거: 고정 GraGOD 포크 `datasets/dataset.py:47-50,64-77`; `models/gdn/model.py:292-316`; `models/predict.py:121-145`; `src/gdn_runner/run_gdn_single.py`; `tests/test_run_gdn_single.py`; `docs/score_interface.md`
- 기록일: 2026-08-17

### D-35
- 결정: 학습은 TSAD와 GraGOD 작업 트리가 모두 clean이고 GraGOD HEAD가 `485e26b0c6b1d63f4f3531c8d05597db82e9db29`일 때만 시작한다. snapshot은 의존성 import와 학습보다 먼저 저장한다. 실행 종료 때 두 HEAD와 clean 상태를 다시 검사한다. Git 조회 실패를 `unknown`으로 바꾸지 않는다.
- 근거: `src/common/verify_run_context.py`; `src/gdn_runner/run_gdn_single.py`; `tests/test_verify_run_context.py`; D-01·D-14·D-32
- 기록일: 2026-08-17

### D-36
- 결정: 실행 입력은 `configs/input_manifest.yaml`의 크기와 SHA-256이 모두 맞아야 한다. GHL·HAI 러너는 검증한 파일 지문이 snapshot 입력에 없거나 형식이 틀리면 학습 전에 멈춘다. label은 길이가 맞는 유한한 0·1만 허용하고 scaler의 반환 배열을 반드시 사용한다. 실행 비율과 seed는 YAML만 읽으며 허용 비율 `5·10·20·50·100`은 파일명·분할·조합 코드가 한 상수를 공유한다. 환경은 package 버전뿐 아니라 저자 `pytorch-lightning` 포크의 설치 URL과 commit도 대조한다.
- 근거: `configs/input_manifest.yaml`; `configs/environment.yaml`; `src/common/verify_input_files.py`; `src/common/verify_run_context.py`; `src/common/experiment_config.py`; `src/data_split/validate_labels.py`; `tests/test_verify_input_files.py`; `tests/test_verify_run_context.py`; `tests/test_experiment_config.py`; `tests/test_load_gdn_inputs.py`
- 기록일: 2026-08-17

### D-37
- 결정: 모든 fit은 학습, train-reference 추론, test 추론 시간을 `timing.json`에 나눠 저장한다. 완전성 검사는 필수 파일과 checkpoint가 존재하며 0바이트가 아닌지도 확인한다. GHL −TOPK는 25시계열×2비율×3seed=150회다. TopK 민감도는 `k={2,5,10}`의 논리 조합 450개이며 k=5 주 실행 150개를 재사용하므로 추가 fit은 300회다. −TOPK의 실제 추가 fit까지 합치면 450회다. 대조 팔은 `GDN_NOTOPK`, `GDN_K2`, `GDN_K10`으로 분리하고 −TOPK snapshot의 사용되지 않는 topk도 기준 YAML 값을 그대로 기록한다.
- 근거: `configs/gdn_hyperparams.yaml`; `experiments/exp02_gdn_ghl/check_controls.py`; `experiments/exp02_gdn_ghl/run_controls.py`; `experiments/exp02_gdn_ghl/check_completeness.py`; `src/gdn_runner/run_gdn_single.py`; `tests/test_gdn_batch.py`; 고정 GraGOD 포크 `models/gdn/model.py:136-159`
- 기록일: 2026-08-17

### D-38
- 결정: 이번 1차 보강에서는 실제 GHL·HAI 학습과 합성 모델 학습을 실행하지 않는다. 단위 테스트, YAML·환경 계약, Python compile, diff 정적 검사까지만 수행한다. 4단계는 변경 commit, clean 작업 트리, 새 계약으로 고친 합성 드라이런을 차례로 통과한 뒤에만 연다.
- 근거: `docs/pre_run_checklist.md`; `docs/NEXT_SESSION_PLAN.md`; 사용자 지시(2026-08-17)
- 기록일: 2026-08-17

### D-39
- 결정: 4단계 전 2차 감사에서는 3a의 Manifest·비율 근거를 봉인 로그로 다시 계산하고 3b의 입력부터 완료 판정까지 전 경로를 추적했다. 3a 수치와 비율 역할은 유지한다. 3b는 네 지점을 보강한다. 실행 전 계약은 `epsilon=0.01`, smoothing 창 4, testnorm 병행을 함께 검사한다. forecast 오차가 비유한 값이면 점수를 저장하지 않으며, HAI embedding이 비유한 값이거나 norm 0이면 정의되지 않은 cosine graph를 저장하지 않는다. 배치는 러너의 종료 검증과 HAI 그래프 저장이 끝난 뒤에만 `COMPLETE` 표식을 쓰고, 표식 없는 산출물은 재개 대상에서 완료로 세지 않는다. 모든 조합이 이미 끝났다면 원본 입력 SHA-256을 다시 계산하지 않는다.
- 근거: `src/common/experiment_config.py`; `src/gdn_runner/run_gdn_single.py`; `src/gdn_runner/extract_adjacency.py`; `src/common/run_completion.py`; exp02·exp03의 `run_batch.py`와 `check_completeness.py`; `tests/test_experiment_config.py`; `tests/test_run_gdn_single.py`; `tests/test_extract_adjacency.py`; `tests/test_gdn_batch.py`; 고정 GraGOD 포크 `models/gdn/model.py:139-144`, `datasets/dataset.py:47-50,64-77`; `docs/superpowers/plans/2026-08-17-stage3-second-audit.md`
- 기록일: 2026-08-17

### D-40
- 결정: 4단계 전 3차 감사에서 3a의 데이터 수치·비율 역할·하이퍼파라미터는 유지했다. 3b 재개 판정은 현재 clean TSAD·GraGOD hash와 snapshot의 두 hash가 같을 때만 `COMPLETE`를 인정하도록 고쳤다. 이전 commit 산출물과 형식이 깨진 snapshot은 재실행 대상으로 돌린다. 완료 검사 CLI와 HAI Jaccard도 같은 신원을 확인하며, Jaccard는 현재 commit의 HAI 조합이 모두 끝나야 계산한다. TopK는 대각선을 가리지 않아 보통 self-edge가 포함되지만 cosine 동률에서는 보장되지 않으므로 “항상 k−1개 + 자기 1개”라는 설명을 폐기하고 실제 추출 edge에서 self-edge를 제거한다. `k=max(5, ceil(0.25N))`의 0.25는 원 논문의 WADI `30/127=23.6%`와 SWaT `15/51=29.4%` 사이에 둔 사전 기준이며 데이터에서 고른 최적값이 아니다.
- 근거: `src/common/run_completion.py`; exp02·exp03의 `run_batch.py`와 `check_completeness.py`; `experiments/exp02_gdn_ghl/run_controls.py`; `experiments/exp03_gdn_hai_seed10/compute_jaccard_agreement.py`; `tests/test_gdn_batch.py`; `docs/gdn_hyperparameter_decisions.md`; 고정 GraGOD 포크 `models/gdn/model.py:136-159`, `models/gdn/modules.py:213-214`; 고정 Lightning `pytorch_lightning/loops/prediction_loop.py:273-274`; `docs/superpowers/plans/2026-08-17-stage3-third-audit.md`
- 기록일: 2026-08-17

### D-41
- 결정: 4단계 전 4차 감사에서 폐기된 `L-W-1` 계산이 GHL·HAI preflight의 정규화 표본 수에 남은 것을 고쳤다. 두 생성식과 봉인 로그 665행은 현행 1-step forecast 계약인 `L-W`로 통일한다. 모든 feasibility 판정은 그대로 통과한다. 실행 시간에는 실제 `accelerator`를 함께 기록하고, HAI Jaccard의 쌍별·요약 CSV 각 행에는 TSAD와 GraGOD commit을 넣어 서로 다른 실행의 표를 구분한다. 실제 입력 길이 23종에서 validation 10%의 부동소수점 ceil과 정확 유리수 계산은 모두 같았으므로 split은 바꾸지 않았다. 비율·seed·채널·topk·window·대조 팔 조합도 유지한다.
- 근거: `experiments/exp01b_ghl_preflight/run_ghl_preflight.py`, `logs/ratio_feasibility.csv`, `ANALYSIS.md`; `experiments/exp01c_hai_preflight/run_hai_preflight.py`, `logs/ratio_feasibility.csv`, `ANALYSIS.md`; `src/gdn_runner/run_gdn_single.py`; `experiments/exp03_gdn_hai_seed10/compute_jaccard_agreement.py`; `tests/test_ghl_preflight.py`; `tests/test_hai_preflight.py`; `tests/test_run_gdn_single.py`; `tests/test_gdn_batch.py`; 고정 GraGOD 포크 `datasets/dataset.py:47-50,64-77`, `gragod/utils.py:83-92`; `docs/superpowers/plans/2026-08-17-stage3-fourth-audit.md`
- 기록일: 2026-08-17

### D-42
- 결정: 4단계 전 5차 감사에서 D-34~D-41의 확정값을 대조한 결과, 같은 파라미터를 되돌린 순환 수정은 없었다. 비율·seed·validation·window·topk는 유지하고, 장시간 배치에서 manifest 검증 뒤 실제 로드 전에 외부 입력이 바뀔 수 있는 경계만 보강한다. 최초 SHA-256 계산 전후의 크기와 `mtime_ns`가 같아야 지문을 발급하며, GHL 주 배치·대조군과 HAI 배치는 새 입력을 읽기 직전과 직후에 같은 상태를 확인한다. 이 검사는 일반적인 파일 교체·수정·삭제를 막되 SHA-256을 로드마다 다시 계산하지 않는다. 새 diff·manifest 불일치·고정 외부 레포 변경·실패 테스트·dryrun 실패가 없으면 3단계 정적 감사를 다시 열지 않는다.
- 근거: `src/common/verify_input_files.py`; `experiments/exp02_gdn_ghl/run_batch.py`; `experiments/exp02_gdn_ghl/run_controls.py`; `experiments/exp03_gdn_hai_seed10/run_batch.py`; `tests/test_verify_input_files.py`; `tests/test_gdn_batch.py`; 고정 GraGOD 포크 `datasets/dataset.py:47-50,64-77`, `models/train.py:119-145`; `docs/superpowers/plans/2026-08-17-stage3-fifth-audit.md`
- 기록일: 2026-08-17

### D-43
- 결정: 4단계 전 6차 감사에서 설정→loader→GraGOD 학습·예측→점수→snapshot→완료·재개 경로를 다시 대조했으나 실행 코드의 새 결함은 재현되지 않았다. 비율·seed·validation·window·topk, `L-W` 정렬과 실험 조합은 유지한다. 계획서에 남은 Manifest 확정 전 HAI 범위 `59~86ch`만 HAI 23.05의 확정값 `86ch`로 바로잡는다. `COMPLETE` 뒤 외부 파일 변조까지 잡기 위한 출력 전체 재검사는 D-37의 범위와 검증 비용 규율을 벗어나므로 추가하지 않는다.
- 근거: `docs/plan_v4.md:159,164,172,225`; `docs/manifest_draft.md:43-66`; `configs/data_preprocessing.yaml`; `configs/gdn_hyperparams.yaml`; `src/gdn_runner/run_gdn_single.py`; `src/common/run_completion.py`; exp02·exp03의 `run_batch.py`와 `check_completeness.py`; `tests/test_gdn_batch.py`; 고정 GraGOD 포크 `datasets/dataset.py:47-50,64-77`, `models/gdn/model.py:100-200,271,304`, `gragod/training/trainer.py:119-138,206-228`; `docs/superpowers/plans/2026-08-17-stage3-sixth-audit.md`
- 기록일: 2026-08-17

### D-44
- 결정: 4단계 전 7차 감사에서도 비율·seed·validation·window·topk와 `L-W` 정렬은 유지한다. 실행 알고리즘의 새 결함은 재현되지 않았다. 대신 재현 경로와 해석 계약의 네 공백을 닫는다. 패치 재검증 문서는 저장소 안의 정확한 경로를 적고, Jaccard의 `analysis/`는 정상 실행 산출물로 Git에서 제외한다. GHL preflight 생성기와 현재 분석의 미확정 W 문구는 D-22의 W=5 확정 상태로 바꾸고 `L-W` 정규화 표본 문구가 재생성 때도 유지되게 한다. GHL 25개는 같은 simulator family의 benchmark task이므로 Wilcoxon·TOST·bootstrap은 GHL 내부 민감도 요약으로만 읽고 제조 공정 모집단으로 일반화하지 않는다.
- 근거: `AGENTS.md`; `configs/environment.yaml`; `experiments/exp00_gragod_recon/PATCH_REVERIFY.md`; `.gitignore`; `experiments/exp03_gdn_hai_seed10/compute_jaccard_agreement.py:94-105`; `experiments/exp01b_ghl_preflight/run_ghl_preflight.py:370-374`, `ANALYSIS.md:47-52`; `docs/plan_v4.md:247-255`; `docs/role_C.md:34-36`; 고정 GraGOD 포크 `datasets/dataset.py:47-50,64-77`, `models/gdn/model.py:100-200,271,304`, `gragod/training/trainer.py:119-138,206-228`; `docs/superpowers/plans/2026-08-17-stage3-seventh-audit.md`
- 기록일: 2026-08-17

### D-45
- 결정: 3단계 변경을 한 commit으로 봉인하고 clean 합성 드라이런을 통과한 뒤, 4단계의 첫 실행은 `GHL series 01·10%·seed 1` 한 건으로 제한한다. 현재 TSAD·GraGOD commit과 snapshot의 두 hash가 같고 필수 산출물과 `COMPLETE`가 모두 있을 때만 성공으로 판정한다. 이 스모크 결과를 확인하기 전에는 GHL 전체 배치를 실행하지 않는다.
- 근거: 사용자 실행 지시(2026-08-17); `experiments/exp00_gragod_recon/dryrun_synthetic.py`; `experiments/exp02_gdn_ghl/run_batch.py`; `experiments/exp02_gdn_ghl/check_completeness.py`; `src/common/run_completion.py`
- 기록일: 2026-08-17
