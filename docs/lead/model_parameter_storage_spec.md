# ER diagram 담당자용 입력 특징·모델 파라미터 명세

기준일: 2026-09-09. 현행 registry와 실제 호출 인자를 대조한 전달 명세다.
모델 값은 현행 코드의 값이며, 아래의 열 분리와 관계는 ER diagram 설계를 위한 제안이다.
모델별 상세 표는 전달 설계다. 2026-09-09에는 입력 특징 네 항목을 실제 SQLite schema와
수집·내보내기 코드에 추가했다. 이 수집 변경은 모델 후보·학습식·선택식을 바꾸지 않는다.
ALoRa는 제외 상태이며 현재 36개 설정·45개 head별 점수를 보존한다.

## 정보를 나누는 기준

모델 파라미터도 분석용 설명 변수로 사용할 수 있다. 입력 데이터에서 계산한 특징과는 별도
묶음으로 저장하고 키로 연결한다. 성능·실측 비용·선택 여부는 실행 후 얻는 값이다.
추천 입력을 구성하는 후속 단계에서도 이 구분을 유지해야 한다.

| 묶음 | 한 행의 의미 | 핵심 항목 | 연결 기준 |
| --- | --- | --- | --- |
| 입력 특징 | 한 CSV의 한 q-prefix | q, 관측 행 수, 센서 수, 상수 채널 수, 채널 std·ACF·상관과 추가 네 특징의 중앙값 | prefix_feature_id |
| 채널 특징 | 한 prefix의 센서 하나 | 채널 순서·이름, mean·std·median·ACF, IQR·차분 Q90/IQR·중앙값 이동/IQR·spectral entropy, 유효값 수와 사유 | prefix_feature_id + channel_index |
| 모델 설정 | 한 모델의 후보 설정 하나 | 탐색 파라미터·고정 파라미터·전후처리·소스 신원 | config_id |
| 실행 기록 | 실제 실행 시도 한 번 | seed, 환경, 실제 학습량, 선택 epoch, 시간·메모리, 상태 | 실행 이력의 run_id. 점수에 연결된 값이 physical_execution_id다. |
| 성능 결과 | 한 CSV/q/config/seed/head의 점수 | VUS-PR, 채점 상태, 평가기·점수 파일 참조 | prefix_feature_id + config_id + seed + score_variant |
| 선택 정책 | 한 실험·정책 종류·조건집단·q의 선택 | 선택 모델·config, head 표, 개발 평균, LOFO와 선택식 버전 | 실험 신원 + analysis_kind + group_id + q |
| 학습된 산출물 | 실제 실행에서 얻은 모델 상태 | 가중치, scaler, PCA 성분, PaAno memory bank | training_files의 checkpoint·scaler_state 등 파일·SHA-256 |

설정의 `epochs=50`과 실행 결과의 `epochs_completed=17`은 서로 다른 값이다.
PaAno의 `memory_fraction=0.1`은 설정이고 `memory_count`는 현재 입력으로 결정되는 실행 값이다.
큰 가중치 배열은 checkpoint 파일을 참조한다. GDN scaler처럼 별도 저장되는 상태 파일도
각각 연결한다.

## 공통 필드와 키

추가 특징의 정확한 열 이름·산식·미정의 조건은 [수집 계약](tuning_feature_capture_prompt.md#최소-필수-특징과-계산-정의)을
따른다. 산식은 `prefix_features.v2`, 추천 DB schema는 3이다. 모델 metadata의 저장 버전과
구분하며 과거 DB를 자동 이관하지 않는다.
추천용 VIEW·CSV에는 원시 진폭 통계를 제외하고 채널별 IQR/std 중앙값과 무차원 특징,
상수·유효 비율·NULL 사유를 연결한다. [단위·검증 계약](tuning_feature_capture_prompt.md#추천-입력의-단위중복결측-처리)을 따른다.

| 항목 | 자료형·범위 | 의미와 규칙 |
| --- | --- | --- |
| 실험 신원 | TEXT | 현재 SQLite는 실험당 하나다. 여러 실험을 합치는 DB에는 experiment_id 등 명시적 범위 키를 추가한다. budget·코드·입력·schema 신원도 연결한다. |
| prefix_feature_id | TEXT | CSV 신원·원본 SHA·채널 순서·q·입력 범위·특징 산식으로 만든 기존 ID를 쓴다. 모델·config·seed와 독립이다. |
| q_percent | INTEGER: 5,10,20,40,60,80,100 | 이용 가능한 정상 prefix 비율이다. |
| training_boundary | INTEGER, 행 | q의 기준인 정상 학습 구간 N이다. 전체 CSV 길이가 아니다. |
| observed_row | INTEGER, 행 | floor(N×q/100). 실제 내부 학습·검증 창 수와 구분한다. |
| input_column | INTEGER, 개 | 입력 센서 수다. scalar 출력의 채널 수를 가져오지 않는다. |
| config_id | TEXT | 모델·소스 commit·checkpoint SHA·설정·전후처리·공통 recipe로 생성한 기존 ID를 유지한다. CSV·q·seed·실측 비용을 넣지 않는다. |
| model / tier / target_use | TEXT | 정확한 등록 이름, t1/t2/t3, training_free/fit_full_prefix/strict_zero_shot. PCA도 target_use는 training_free지만 평가 입력 fit 출처를 별도로 보존한다. |
| deterministic | BOOLEAN | 현행 PaAno·GDN은 false, 나머지는 true다. 개발 seed는 전자는 0·1·2, 후자는 0이다. |
| physical_execution_id | TEXT | 여러 논리 q·head가 한 실제 실행을 참조할 수 있다. 실행 횟수·시간을 q나 head 수만큼 합산하지 않는다. |
| score_variant | TEXT | 실제 점수 head. TSPulse는 time/fft/pred/ensemble, 그 밖의 현행 결과는 기존 빈 문자열을 유지한다. |
| source_commit / source_checkpoint_sha256 | TEXT | 원본 구현과 사전학습 가중치의 신원이다. 이번 실행이 만든 학습 checkpoint의 SHA와 구분한다. |
| recipe / checkpoint config / settings_json | JSON 또는 참조 | 전체 원설정을 보존한다. checkpoint_config_sha256가 있는 모델은 함께 연결한다. |

숫자·불리언은 문자열로 저장하지 않는다. 알 수 없거나 정의되지 않은 값은 NULL과 사유를
함께 보존한다. 창·문맥·patch의 단위는 시점 수다. 이를 초·분으로 바꾸려면 파일별로 확인한
sampling interval이 필요하며 값을 추정하지 않는다.

## 모델별로 분리할 파라미터

아래 이름은 현행 `hyperparameters` 키다. 역할은 탐색값·고정값을 구분하며, 레시피와 실행
제어는 별도로 표시했다. 모델별 상세 열은 `config_id`로 공통 설정과 연결한다.

### One-Liners — 16개 설정

| 모델 | 탐색값 | 고정값과 자료형 |
| --- | --- | --- |
| MWVAR | window INTEGER: 5,10,32,50,60,64,96,100,256,512,1024 | centered BOOLEAN=true, ddof INTEGER=1 |
| SQDIFF_LAST1 | 없음 | lag INTEGER=1, window INTEGER=2, correction REAL=2.0 |
| SQDIFF_LAST3 | 없음 | lag INTEGER=3, window INTEGER=4, correction REAL=1.3333333333333333 |
| SQDIFF_CENTERED5 | 없음 | window INTEGER=5, centered BOOLEAN=true, correction REAL=1.25 |
| MWVAR96_SQDIFF_LAST3 | 없음 | variance_window INTEGER=96, difference_window INTEGER=4, difference_centered BOOLEAN=false, difference_correction REAL=1.3333333333333333 |
| MWVAR96_SQDIFF_CENTERED5 | 없음 | variance_window INTEGER=96, difference_window INTEGER=5, difference_centered BOOLEAN=true, difference_correction REAL=1.25 |

window는 이동 통계 창 길이이며 앙상블에는 분산 창과 차분 창이 따로 있다.
SQDIFF_LAST3의 lag=3은 이전 3개 관측의 평균을 사용하는 의미다. 단일 시점 x[t−3]과의
차분으로 해석하지 않는다. One-Liner 여섯 등록 이름을 model 열에서 구분한다.

### PCA_LEGACY — 4개 설정

| 원래 키 | 자료형·값 | 역할·해석 |
| --- | --- | --- |
| n_components | REAL: 0.25,0.5,0.75 또는 JSON null | 탐색값. 비율은 설명 분산 목표이며 정수 성분 개수가 아니다. null은 전체 성분 선택이다. |
| window | INTEGER=100 | 고정 창 길이 |
| zero_pruning | BOOLEAN=true | 고정 전처리 |

조회 열은 `n_components_mode`를 variance_fraction/all로 나누고,
`n_components_fraction`에 비율을 둔다. all일 때만 비율 열은 NULL이다. 미수집·해당 없음과
혼동하지 않도록 한다. 원본 `n_components` 값은 settings_json에 그대로 남긴다.
실제로 fit한 성분 개수와 제거·유지한 window feature 수는 실행 결과에 속한다.

### PaAno — 9개 설정

| 키 | 자료형·값 | 역할 |
| --- | --- | --- |
| patch_size | INTEGER: 32,64,96 | 탐색값, 시점 수 |
| learning_rate | REAL: 0.001,0.0001,0.00001 | 탐색값 |
| iterations | INTEGER=100 | 고정 반복 한도. epoch와 같은 열에 넣지 않는다. |
| batch_size | INTEGER=512 | 고정 학습 배치 |
| weight_decay | REAL=0.0001 | 고정값 |
| memory_fraction | REAL=0.1 | 고정 memory 비율 |
| memory_seed | INTEGER=42 | 고정 memory 구축 seed. 실행 seed와 구분한다. |
| neighbors | INTEGER=3 | 점수 계산에 쓰는 최근접 memory 수 |
| use_revin | BOOLEAN=true | 고정값 |
| memory_policy | TEXT=official_minimum | 레시피 preprocess_recipe.memory_count와 adapter 인자. hyperparameters의 별도 탐색 축이 아니다. |

모델 구조·학습법을 설명할 고정 요소는 optimizer=AdamW, embedding_dimension=64,
CNN 채널 폭 [128,256,128,64], kernel 길이 [7,5,3,3], projection 폭 [256,256]이다.
이 값은 현행 소스에 고정되어 있으며 조회용으로 추출할 때 소스 신원을 연결한다.
후보 hyperparameters에 새 축으로 추가하거나 config_id를 바꾸지 않는다.

최신 공식 저장소에 별도 HPO 탐색 루프가 없어 연결 논문의 patch_size와 learning_rate
전체 곱 9개를 유지한다. 학습 patch 수를 P라 하면 실제 memory 개수는 공식 코드의
max(min(500,max(1,P−1)),min(round(0.1×P),P−1))이다. P=1,000이면 500개다.
실제 개수는 실행 로그에서 읽고 config 값으로 복제하지 않는다. paper_fraction은 명시적으로
지정한 과거 경로와 checkpoint 복원에만 남긴다.

### GDN — 정해진 세 조합

다음 열을 각각 분리하되 세 행의 조합을 그대로 보존한다. 각 열의 고유값을 다시 곱해
후보를 늘리면 현행 튜닝 풀이 달라진다. optimizer_betas 배열은 조회 열 beta1·beta2로 나눈다.

| 조합 | embedding INTEGER | hidden INTEGER | topk INTEGER | batch_size INTEGER | epochs INTEGER | patience INTEGER | validation_ratio REAL | beta1 REAL | beta2 REAL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 공식 run.sh | 64 | 128 | 5 | 32 | 30 | 15 | 0.2 | 0.9 | 0.999 |
| 논문 SWaT 주요 값+공식 코드 보완 | 64 | 64 | 15 | 32 | 50 | 15 | 0.1 | 0.9 | 0.999 |
| 논문 WADI 주요 값+공식 코드 보완 | 128 | 128 | 30 | 32 | 50 | 15 | 0.1 | 0.9 | 0.999 |

최신 공식 저장소에 전체 HPO 풀이 없어 승인한 세 조합을 유지한다. 세 조합 모두
공식 train.py의 고정 patience=15와 Adam 기본 betas=(0.9,0.999)를 적용한다.

| 고정 키 | 자료형·값 | 의미 |
| --- | --- | --- |
| window / stride | INTEGER=5 / INTEGER=1 | 예측 입력 창·이동 간격 |
| learning_rate / weight_decay | REAL=0.001 / REAL=0.0 | optimizer 설정 |
| out_layer_num | INTEGER=1 | 출력 계층 수 |
| graph_heads / graph_dropout | INTEGER=1 / REAL=0.0 | 그래프 attention head 수·dropout |
| output_dropout | REAL=0.2 | 출력 dropout |
| optimizer | TEXT=Adam | optimizer 종류 |

현재 out_layer_num=1에서 hidden은 사용되지 않는다. 원값을 보존하고 조회용
`hidden_is_active=false`를 함께 명세한다. embedding·topk·학습 설정의 차이는 유효하다.
topk는 채널 수 조건을 만족해야 하며 불가능한 후보는 제외 사유와 함께 남긴다.
epochs는 상한이다. 실제 epochs_completed·selected_epoch·내부 검증 창 수는 실행 기록에 둔다.

### TimeRCD — 1개 설정

| 키 | 자료형·값 | 역할 |
| --- | --- | --- |
| context_length | INTEGER=5000 | 고정 추론 문맥 길이 |
| checkpoint_variant | TEXT=multi | 고정 사전학습 checkpoint 종류 |
| score_head | TEXT=probability | 고정 출력 종류 |

사전학습 가중치의 revision·파일·SHA와 checkpoint 설정 SHA도 연결한다.
이번 튜닝의 학습률·학습 epoch는 해당하지 않는다. 0을 넣어 학습한 모델처럼 표시하지 않는다.

### TSPulse — 3개 설정·12개 head 점수

| 키 | 자료형·값 | 역할 |
| --- | --- | --- |
| aggregation_window | INTEGER: 64,96,128 | 공통 창 선택 후보 |
| context_length | INTEGER=512 | 고정 입력 문맥 |
| patch_size | INTEGER=8 | 고정 patch 길이 |
| heads | TEXT 목록: time,fft,pred,ensemble | 고정 출력 목록. attention head 수가 아니다. |
| inference_batch_size | INTEGER=128 | 레시피·실행 제어. 학습 batch나 탐색 축으로 해석하지 않는다. |
| native_smoothing_window | INTEGER=8 | 레시피. aggregation_window와 별개의 time·fft 후처리 창이다. |

heads는 설정이 지원하는 출력 목록이다. 개별 점수 행의 `score_variant`와 최종 정책의
선택 head를 별도로 둔다. `family_selected`는 정책의 별칭이며 다섯 번째 물리 head가 아니다.

## 숫자 파라미터 외에 필요한 모델 요소

같은 window·학습률이어도 전처리·checkpoint 선택·후처리가 다르면 다른 설정이다.
아래 항목은 전체 recipe를 보존한 상태에서 조회할 핵심 열 또는 참조로 제공한다.

| 모델 | 함께 전달할 요소 |
| --- | --- |
| One-Liners | centered 여부·경계 채움·채널 max. 앙상블은 전체 평가 입력의 component min-max와 component max |
| PCA | 전체 평가 입력 fit, window 정규화·StandardScaler, zero pruning, 가중 성분 거리 |
| PaAno | native RevIN, 학습 loss 기반 checkpoint 선택, official_minimum, top-3 cosine·overlap 평균, 학습 말미 문맥 연결 |
| GDN | prefix fit MinMax, window 내부 검증 분할·검증 loss checkpoint, 고정 topk, 전체 평가 오차 median/IQR·trailing4·채널 max |
| TimeRCD | multi checkpoint, 전체 평가 채널 z-score, context 블록·말미 padding·출력 집계 |
| TSPulse | 전체 평가 StandardScaler·model RevIN, head별 경계 복원·정규화·smoothing·ensemble, 두 단계 선택식 버전 |

가중치 학습 여부와 평가 입력 통계 사용 여부도 분리한다. training_free나 strict_zero_shot이라는
이름만 보고 통계 fit이 없다고 추정하지 않는다. 소스 commit·preprocess_recipe·common_recipe의
원본과 연결해 위 설명을 확인할 수 있어야 한다.

## 실행 뒤에 저장할 값과 선택 근거

| 대상 | 설정과 분리할 값 | 현행 출처·추가 명세 범위 |
| --- | --- | --- |
| PaAno | training_patch_count, memory_count, effective_batch_size, optimizer_updates, training_examples_seen, selected_iteration, loss_history | training_log에 기록된다. loss_history는 iteration별 목적별 loss·가중치·학습률이며 실제 memory bank는 checkpoint다. |
| GDN | epoch_cap, epochs_completed, selected_epoch, optimizer_updates, best_validation_loss, loss_history, 실제 내부 검증 범위 | training_log와 model_internal_validation에 기록된다. loss_history는 epoch별 학습 batch MSE 합·평균·검증 loss다. |
| PCA | 제거·유지한 window feature 수, 실제 성분 수 | 앞의 두 값은 metadata에 있다. 실제 성분 수는 fitted PCA에 있으며 독립 DB 열로 추출하는 일은 추가 구현 대상이다. |
| 공통 자원 | 학습·추론 시간, peak GPU/RAM과 측정 범위, 실행 장치·환경, 재시도 상태 | 실행 증거를 참조한다. 모델별로 측정되지 않은 값은 사유와 함께 NULL로 둔다. |
| 성능 | 파일별 VUS-PR, evaluator SHA, ell_max 신원, 완료 상태 | 기존 결과 원표·ledger를 연결한다. seed 평균·family 평균과 구분한다. |
| 선택 | model_ratio/tier_adaptive, q, group_id, 지원 파일·family, 선택 config, 평균 종류·값, LOFO fold | 선택 JSON·원표가 출처다. 별도 DB 표로 정규화하려면 ERD에 관계를 명시한다. |

TSPulse는 q별 공통 창의 후보 점수·집계식 ID와 데이터셋별 head 표를 보존한다. head 표는
정책과 family의 조합으로 연결하며 미관측 family의 기본 head=time도 기록한다.
LOFO에는 학습 family·holdout family와 그 fold에서 다시 고른 창·head를 연결한다.
전체 개발 패널의 선택과 fold별 선택을 덮어쓰지 않는다.

배치 추론·거리 분할 크기와 CPU worker 수는 실행 제어다. 예를 들어 PaAno/PCA의 거리
분할, TimeRCD의 attention 분할, TSPulse의 추론 batch는 학습 파라미터와 구분한다.
L4 24GB·CPU 8개는 대상 자원이며 실제 RAM·backend·환경은 원격 실행에서 확인해 기록한다.

## 열 분리 방식과 담당자 인수 기준

현재 DB의 `model_configs`는 config_id·model·target_use·settings_json을 저장한다.
각 후보의 확정 파라미터는 `settings_json.hyperparameters`에 이미 있다. 고정값·후보값을
합친 이 경로에서 추출하며 grid나 설명문을 다시 해석하지 않는다.

기본안은 전체 JSON을 원본으로 보존하고 모델군별로 타입이 정해진 열을 제공하는 조회 view를
두는 것이다. JSON 안의 learning_rate를 매번 직접 찾지 않고 숫자 열로 비교할 수 있다.
ERD에서 물리 열 저장을 요구한다면 `config_id`를 PK/FK로 쓰는 모델군별 1:1 상세 테이블에
같은 열을 저장한다. 어느 방식이든 추출값은 원본과 일치해야 하며 독립적으로 수정하지 않는다.
모든 모델의 파라미터를 한 표에 모아 해당 없는 열을 채우거나 모든 값을 문자열로 저장할
필요는 없다. 분석용으로는 입력 특징·설정·결과를 조인한 넓은 조회표를 제공한다.

파라미터 이름을 조회용으로 바꾸거나 beta 배열을 둘로 나눠도 config_id를 다시 계산하지
않는다. 기존 canonical JSON과 원본 source·checkpoint·recipe 신원을 그대로 사용한다.
모델군별 상세 관계를 택하면 해당 모델의 상세 행은 하나이며 다른 모델의 상세 행은 없다.

현재 성공 점수의 physical_execution_id는 metadata.execution_attempt.run_id를 참조한다.
재시도마다 run_id가 달라지므로 동일한 논리 trial의 여러 시도는 실행 이력에서 묶는다.
새 DB에서도 이 관계를 보존하고 이미 있는 run_id와 같은 의미의 ID를 중복해서 만들지 않는다.

ER diagram 담당자에게 이 명세와 현행 registry를 함께 전달한다. 담당자는 필드명·자료형·
단위·NULL 의미·PK/FK·한 행의 범위를 명시하고, 작은 예시에서 다음을 확인할 수 있게 한다.

- PaAno 설정의 memory_fraction과 실행의 memory_count를 따로 조회한다.
- GDN 세 조합을 그대로 복원하고 비활성 hidden을 식별한다.
- PCA의 전체 성분 모드와 결측값을 구분한다.
- 같은 실제 실행을 여러 q·head가 참조해도 비용을 한 번만 센다.
- TSPulse의 모든 원점수와 q별·fold별 선택 근거를 다시 연결한다.

실제 저장 구현은 확정 ERD와 매핑한 뒤 진행한다. 기존 성능표 11열과 네 묶음 내보내기는
현재 계약으로 남기고, 추가 파라미터 조회·내보내기의 형식과 schema 버전은 별도로 명세한다.
큰 점수·checkpoint 파일은 DB와 함께 추적할 보관 위치·크기·SHA-256·경로 이전 규칙을 둔다.

## 근거와 현재 상태

- [현행 모델 registry](../../configs/model_registry.yaml)
- [실제 adapter 호출 인자](../../src/common/run_registered_model.py)
- [config 식별자](../../src/common/build_config_id.py)
- [현재 SQLite 저장 코드](../../tests/ghl_main/store_recommendation_evidence.py)
- [기존 특징·저장 계약](tuning_feature_capture_prompt.md)
- [Lightning 실행 안내](lightning_studio.md)

설정 선언·호출부·저장 원본과 문서의 값·키를 정적으로 대조했다. 테스트·Python·YAML 파싱·
데이터·모델 실행은 하지 않았다. 현재 구현·원격 검증 전 게이트를 유지한다.
