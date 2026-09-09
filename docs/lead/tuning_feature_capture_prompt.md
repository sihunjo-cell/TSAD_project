# 전면 재튜닝 전 추천용 정보·SQL 저장 코드 수정 프롬프트

갱신일: 2026-09-09

2026-09-09 사용자 요청으로 기존 특징에 IQR, 차분 절댓값 Q90/IQR, 전후반 중앙값
이동/IQR, spectral entropy를 추가한다. 채널별 값과 유효 채널 중앙값을 SQLite 원표와
CSV에 저장한다. extractor는 `prefix_features.v2`, 추천 DB schema는 3이다. 기존 기본 표시
열 뒤에 추가하며 성능표 11열은 유지한다. 이 변경은 모델의 학습식·후보·선택식을 바꾸지 않는다.
후속 요청에 따라 단위 영향을 받는 원시 진폭 통계를 제외한 `recommendation_inputs` VIEW와
CSV를 추가한다. 기존 단일 진입점의 DB 생성·적재·백업·인수 절차에서 함께 만든다.

과거 튜닝 결과는 폐기하고 공개 원논문 코드의 후보와 모델별 절차로 다시 실행한다.
2026-09-07 사용자 지시로 내부 검증·checkpoint 선택·전체 평가 통계와 native 후처리까지
포함한다. 기존 full-prefix 고정과 공유 학습보다 이 지시가 우선한다. 이 문서의 수집·저장
계약과 [저장 형식(toBigs).xlsx 첨부본](tuning_storage_format.xlsx)은 유지한다.
최신 사용자 지시로 ALoRa를 제외한다. 남은 공식 논문·구현 기반 풀은 36개 설정·45개 head별 선택 항목이며 추가 축소하지 않는다.

## 구현 명령

### 목적과 범위

새 전면 튜닝에서 파일별·q별 입력 특징과 전체 후보의 평가 결과를 확보하고, 첨부 엑셀의
네 묶음인 성능 결과·입력 특징·채널별 특징·모델 설정을 SQL로 연결해 관리하도록 구현하라.
같은 입력 특징을 모델·config·seed마다 계산하거나 중복된 학습 행 수를 저장하지 마라.

이 명령은 코드·설정·필요한 문서·회귀 검사 작성까지다. 실제 대규모 튜닝은 구현과 원격 검증
이후 별도 명령으로 수행한다. 로컬 테스트·Python 구문 검사·YAML 파싱·원본 데이터 스캔·
모델·채점·그림 실행 금지는 유지한다. 기존 미커밋 코드는 보존하고 관련 부분만 수정한다.
추천기 학습·기업 조회·거리·가중치·순위·지원 상한·GHL·HAI 최종 실행은 이번 범위가 아니다.

`AGENTS.md`, `docs/lead/plan_v5.md`, `docs/lead/decisions.md`,
`docs/lead/process_0_preverify.md`, `docs/lead/next_session.md`를 읽고 현재 게이트를 따른다.

### 폐기 결과와 새 튜닝의 분리

과거 80/20·후보 풀·scaler로 얻은 점수·ledger·선택표·학습 checkpoint·scaler·완료 영수증·
VUS 캐시를 새 실행에 승계하지 않는다. Downloads의 `dev18_tuning_review_bundle`을
복구·보완·이식하지 마라. 과거 성능값이나 우승 모델이 유지되는지도 검사하지 않는다.
새 `full_prefix_v2` 진입점은 과거 결과가 없어도 작동하며, 옛 파일이 남아 있어도 읽거나
봉인하지 않아야 한다. 선택적 baseline SHA 수집·종료 시 재검사 등 잔존 연결도 제거한다.

원본 데이터·manifest·기존 데이터 감사·봉인한 공식 source·공식 사전학습 가중치는 유지한다.
신원이 같은 원본을 불필요하게 다시 감사하지 않는다. ell_max도 승인된 산식·입력 신원을
확인해 사용하며 폐기한 성능값을 가져오지 않는다. 새 source·recipe·budget·환경으로 얻은
완료 기록만 재개에 사용한다. 사용자가 폐기한 과거 결과·예산·선택표를 복구하거나 새 실행에
재사용하지 않는다. 첨부 저장 양식은 폐기 대상 튜닝 결과와 무관한 입력 문서다.

### 유지할 튜닝 계약

| 항목 | 계약 |
| --- | --- |
| 데이터·q | Dev18 원본 18개 CSV와 순서를 유지한다. q={5,10,20,40,60,80,100}. 지정 정상 학습 경계 N에서 앞 floor(N×q/100)행을 사용한다. |
| 분할 | 현재 prefix 전부를 모델에 전달한 뒤 공식 내부 절차를 따른다. GDN의 window 검증 분할과 모델별 checkpoint 선택 근거를 기록한다. |
| PaAno | native RevIN. 과거 공통 외부 MinMax를 적용하지 않는다. |
| GDN | 현재 prefix의 fit-only MinMax, 공식 window 검증 분할·조기 종료·최저 검증 loss checkpoint와 전체 평가 채널 오차 통계의 native 점수를 사용한다. |
| PCA·training-free | PCA는 공식 전체 평가 입력 fit·zero pruning을 복원한다. One-Liners의 공개 창·차분·앙상블을 포함한다. |
| Tier 3 | 가중치 학습 없이 공식 전체 평가 입력 통계와 후처리를 사용한다. TSPulse ensemble을 포함한 네 head를 주후보로 둔다. |
| 후보·선택 | 최신 registry의 전체 후보와 주/보조 head, CSV/q별 가능 후보, model_ratio·tier_adaptive·family·LOFO·membership 계산을 유지하고 새 점수로 다시 산출한다. |

과거 152개 설정·156개 주 선택 항목은 폐기한 구성이다. 실제 후보·실행·채점 수는
확정 registry·CSV/q별 feasibility·seed·head로 다시 봉인한다. 모든 CSV에서
가능한 설정만 남기는 교집합을 만들지 않는다. source·실제 호출 인자·분할·scaler 범위를
대조하고 공식 절차와 다른 임의 기본값을 제거한다. 수집 기능 때문에 입력·RNG·모델 설정·
점수·선택식이 달라지면 안 된다. 논문에 없는 시간 단축은 별도 요청 후 검토한다.

### 첨부 엑셀의 네 묶음과 열 계약

첨부본은 Sheet1의 B6:L6, B10:I10, B14:G14, B24:D24에 열을 정의한다.
빈 행과 설명 행은 서식이며 데이터가 아니다. 아래 명세도 본문에 포함하므로 구현이
사용자의 Downloads 경로나 실행 중 엑셀 읽기에 의존하지 않아야 한다.

| 묶음 | 기본 표시 열과 순서 | 한 행의 의미 |
| --- | --- | --- |
| 성능 결과 | CSV_file, q_percent, training_boundary, observed_row, input_column, prefix_feature_id, model, config_id, seed, status, vus_pr | 개별 CSV/q/config/seed의 한 점수 head 결과 |
| 입력 특징 | prefix_feature_id, csv_file, q_percent, observed_row, 상수채널수, 채널std median, 채널 ACF lag1 median, 채널간 절대 상관 중앙값 | 개별 CSV/q의 입력 특징 |
| 채널별 특징 | prefix_feature_id, channel(col), mean, std, median, ACF lag1 | 해당 prefix의 센서 하나 |
| 모델 설정 | config_id, model, 설정 예시 | 한 config의 전체 설정. 출력에는 예시 일부가 아니라 실제 전체 설정을 넣는다. |

사용자의 열 순서와 이름을 기본 표시 계약으로 유지한다. 기술적인 SQL 이름은 내부에서
통일하되 CSV_file/csv_file 대소문자 차이와 한국어 이름은 조회·내보내기의 alias로 맞춘다.

| 표시 열 | 의미·SQL 내부 대응 |
| --- | --- |
| training_boundary | 비율의 기준인 지정 정상 학습 구간 N. 평가 tail을 포함한 CSV 전체 행 수가 아니다. |
| observed_row | 현재 prefix의 관측 행 수 n. 모델에 전달한 범위이며 실제 내부 학습·검증 창 수와 구분한다. |
| input_column | loader 입력 센서 수 d. scalar 점수 metadata의 channel_count=0을 가져오지 않는다. |
| prefix_feature_id | 같은 CSV/q 입력 특징을 참조하는 ID. model/config/seed와 독립이다. |
| 상수채널수 | constant_channel_count |
| 채널std median | channel_std_median |
| 채널 ACF lag1 median | channel_acf_lag1_median |
| 채널간 절대 상관 중앙값 | absolute_correlation_median |
| channel(col), ACF lag1 | channel_name, acf_lag1. 내부 채널 키는 순서 있는 스키마의 채널 index를 쓴다. |

q_percent는 5·10·20처럼 백분율 숫자로 저장한다. 행·채널 수는 INTEGER, 통계·점수는 REAL,
ID·이름·상태는 TEXT, 미정의 값은 SQL NULL이다. 문자열 "NULL"이나 0으로 채우지 않는다.
입력 특징에 VUS·선택 여부·실측 비용·평가 라벨 통계·미관측 미래 구간 통계를 섞지 않는다.

### 중복 행 수·과거 설정 제거

새 full-prefix 저장 계약에는 `fit_rows`, `fit_count`, `validation_rows`,
`validation_count`, `val_rows`, `val_count`를 두지 않는다. `available_count`나
`observed_prefix_count`도 `observed_row`와 같은 값을 별도 열로 복제하지 않는다.
추천 DB·내보내기뿐 아니라 새 full-prefix metadata/snapshot의 중복 필드도 같은 원칙으로
정리한다. 감사용이라는 이유로 동일한 행 수와 항상 0인 validation 행 수를 다시 추가하지 마라.

실행기 내부의 분할 검증 변수·feasibility 계산은 유지한다. 실행 직전에 학습형에 전달하는
prefix 범위와 모델별 공식 내부 절차를 검사한다.
이 규칙은 실험 recipe에 한 번 기록한다. training-free·strict zero-shot은 config의 기존
`target_use`로 구분하며, observed_row를 실제 학습량이라고 해석하지 않는다.

기존 strict schema·writer·reader·snapshot·완료 영수증이 제거할 필드를 요구하면 새
full-prefix 저장 schema의 버전을 올리고 해당 호출·검증을 함께 맞춘다. 단순히 키만 지워
검증을 깨뜨리지 말고 과거 봉인 파일을 변조하지도 않는다. 이 저장 변경으로 모델 본체나
선택 공식을 다시 작성하지 않는다. window·patch 개수, 반복 학습 표본 노출량, 실제 실행한
epoch/update는 원시 행 수와 다른 실행 증거이므로 필요 정보로 유지한다.

### 입력 특징은 같은 CSV/q에서 한 번만 계산

사용자가 말한 "첫 row에서 산출하고 이후 복붙"은 동일 CSV/q를 처음 준비할 때 특징을
한 번 계산하고, 이후 모델·config·seed·head 결과가 저장된 값을 참조하거나 복사한다는 뜻이다.
첫 성능 결과가 성공할 때까지 미루지 말고 모델 배치 전에 입력 특징을 준비한다.
모든 후보가 제외된 CSV/q에도 특징 기록은 남긴다.

- 파일명·N·d·순서 있는 채널 스키마 등 파일 공통 정보는 기존 manifest에서 가져온다.
- 상수 채널 수·각 채널 mean/std/median/ACF와 그 요약·상관계수는 CSV/q마다 한 번 계산한다.
- 같은 q에서 seed나 모델이 바뀌어도 extractor·median·ACF·상관 계산을 다시 호출하지 않는다.
- q나 실제 입력 범위가 달라지면 별도 특징을 계산한다. q20의 값을 q40·q100에 복사하지 않는다.
- feature 계산은 loader의 float32 입력 중 현재 prefix만 받으며 모델별 scaling 전 값으로 한다.
  float64로 통계를 계산하고 입력 배열·전역 RNG를 변경하지 않는다.
- DB 원표에는 같은 특징을 한 번 저장한다. 펼친 조회·내보내기에 반복해서 보이는 값은
  JOIN이나 저장값 복사로 채운다. 첫 행만 채우고 나머지를 비워 두거나 위 행을 의미 없이
  forward-fill하지 않는다. feature 계산과 정렬·출력 순서를 결합하지 마라.

정상 준비 시 논리 입력 기록은 18×7=126개이며 채널 원표는 7×각 CSV의 d 합계다.
서로 중첩된 q 기록을 126개의 독립 데이터셋으로 취급하지 않는다. 파일 하나씩 처리하고
작은 통계만 유지한다. 원시 prefix 126개·전체 ACF 배열·상관행렬·model별 입력 복사본을
동시에 보관하거나 DB에 넣지 마라.

prefix_feature_id와 캐시에는 source SHA·ordered schema SHA·logical q·정확한 원시 범위·
입력/계산 dtype·extractor 산식/인자 버전을 반영한다. 생성 시각·실행 seed는 넣지 않는다.
같은 새 실험을 재개하면 이미 검증해 저장한 특징은 읽고 누락분만 계산한다. 입력·범위·
schema·산식이 바뀌면 이전 값을 재사용하지 않는다. 원본 SHA 확인은 기존 절차를 재사용하고
후보마다 재계산하지 않는다. 별도 캐시 서버·전역 대형 캐시·새 캐시 프레임워크는 만들지 않는다.

### 최소 필수 특징과 계산 정의

첨부의 요약 네 항목과 채널별 mean/std/median/ACF lag1에 아래 네 특징을 더한다.
나머지 Q01~Q99·꼬리 폭·첫 ACF peak·요약 Q10/Q90 확장은 보류한다. 이번에 추가한
산식은 추천 자료의 프로젝트 정의이며 공식 모델의 튜닝 파라미터가 아니다. 이 목록만으로
추천 성능이나 지원 상한을 보장하지 않는다.

| 항목 | 계산·유효 조건 |
| --- | --- |
| 채널 mean/std/median | 현재 prefix의 유한값으로 계산한다. std는 ddof=0, median은 linear quantile의 Q50. 유한값이 없으면 NULL이다. |
| 상수 여부·개수 | 유한값이 있고 유효 고유값이 하나인 채널을 상수로 센다. IQR 0을 상수와 같게 보지 않는다. 유효값이 없는 채널은 판정 불가다. |
| 채널 ACF lag1 | n≥2인 전부 유한한 비상수 채널에 계산한다. 분자는 인접한 중심화 값의 곱의 합, 분모는 전체 중심화 제곱합이다. lag별 보정·임의 epsilon·period fallback은 없으며 ACF 전체 lag·peak는 계산하지 않는다. |
| 채널std median·채널 ACF lag1 median | 해당 채널 통계의 유효값 사이 Q50. 대상이 없으면 NULL이다. 서로 다른 CSV를 섞지 않는다. |
| 채널간 절대 상관 중앙값 | 전부 유한한 비상수 채널의 서로 다른 쌍에 대한 Pearson 절댓값을 [0, 1]로 제한한 뒤 Q50을 계산한다. 대각·중복 쌍은 제외하고 유효 쌍이 없으면 NULL이다. |
| interquartile_range | 유한값의 linear Q75−Q25. 유한값이 없으면 NULL이며 0은 유효한 값이다. |
| difference_q90_iqr_ratio | 인접 시점 차분 절댓값의 linear Q90을 IQR로 나눈다. n≥2, 전체 값 유한, IQR>0일 때만 계산한다. |
| median_shift_iqr_ratio | floor(n/2)에서 나눈 앞·뒤 구간의 중앙값 차이 절댓값을 IQR로 나눈다. n≥4, 전체 값 유한, IQR>0일 때만 계산한다. |
| spectral_entropy | 평균을 뺀 채널의 rfft에서 DC를 제외한 전력을 확률로 바꾸고 Shannon entropy를 log(전체 비-DC bin 수)로 나눈다. n≥4인 전부 유한한 비상수 채널에 계산한다. 창 함수·padding·양의 주파수 배수 보정은 쓰지 않으며 수치 오차는 [0,1]로 제한한다. |

새 특징마다 `{field}_reason`을 채널 원표에 저장하고, 요약에는 `channel_{field}_median`,
`channel_{field}_median_reason`, `channel_{field}_valid_channel_count`를 둔다. 시간 특징은
결측을 제거해 시간축을 붙이지 않는다. 최소 행 수, 비유한 입력, IQR 0 또는 상수 여부 순서로
미정의 사유를 정하며 분모에 epsilon을 더하지 않는다. FFT 전력은 magnitude를 최대값으로
나눈 뒤 제곱해 계산한다. FFT 배열 자체는 저장하지 않는다.

이전 산식이나 schema의 DB는 덮어쓰거나 자동 이관하지 않는다. 새 코드·특징 산식·예산을
봉인한 DB에서 수집하며, 동일 버전의 재개에는 저장값을 재사용한다. 원본에서 특징만 다시
계산하는 일은 모델 재학습과 별개다. 모델 metadata의 `full_prefix_storage.v1`은 이번
추천 DB schema 번호와 다른 계약이므로 유지한다.

각 채널의 유효값 수·상수 판정, 요약의 유효 채널/쌍 수와 미정의 사유는 값을 해석할 최소
보조 정보로 연결한다. 부분적으로 판정 불가인 상수 개수를 전체 채널의 확정 개수로
오인하지 않게 판정 가능 채널 수도 남긴다. 이를 위해 모든 품질 descriptor를 다시 늘리지 않는다.
계약상 NULL은 수집 실패와 다르며 비유한 계산값도 NULL과 사유로 처리한다.
통계용 유효 채널 선별은 모델 입력 채널 삭제·보간·정제와 연결하지 않는다. 기존 loader의
입력 거절 조건과 모델 채널 정책은 유지한다.

### 추천 입력의 단위·중복·결측 처리

mean·median은 센서의 원점과 단위에, std·IQR은 단위에 따라 달라진다. 서로 단위가 다른
센서의 std나 IQR 중앙값을 추천 모델에 그대로 넣지 않는다. 원시 통계는 감사용으로 보존하고
`recommendation_inputs` VIEW에서 추천 입력을 구분한다. 새 계산은 저장한 채널 통계만 쓰므로
원시 파일을 다시 읽지 않는다.

| 구분 | 추천용 조회 계약 |
| --- | --- |
| 기본 특징 | observed_row·input_column, 상수 채널 비율, ACF lag1·절대 상관 중앙값, 채널별 IQR/std의 중앙값, 차분 Q90/IQR·중앙값 이동/IQR·spectral entropy의 중앙값 |
| IQR/std | 채널마다 먼저 나눈 뒤 유효 비율들의 중앙값을 계산한다. median(IQR)/median(std)는 쓰지 않는다. std=0이면 제외하고 IQR=0·std>0은 0이다. |
| 품질 정보 | 유한값 수/(n×d), 판정 가능 채널 수/d, 특징별 유효 채널 수/d, 유효 상관 쌍 수/[d×(d−1)/2], 기존 개수와 NULL 사유를 함께 출력한다. 분모가 0이면 NULL이다. |
| 문맥·연결 | prefix ID·CSV ID·파일명·series·family·q·training_boundary는 연결과 검증용이다. family는 봉인한 manifest에서 받아 함께 저장한다. 기본 추천 특징에 포함하지 않는다. n=floor(N×q/100)이므로 n·N·q를 무조건 함께 넣지 않는다. |

상수 채널 비율은 상수 수/판정 가능 채널 수다. 판정 가능 비율도 같이 봐야 결측이 많은
입력을 정상으로 오해하지 않는다. 채널 요약의 NULL 사유와 상세 채널의 미정의 사유는
`channel_features`에서 연결하며 NULL을 0으로 바꾸지 않는다.

위 채널 특징은 각 채널을 서로 다른 a×x+b(a≠0)로 바꿔도 정확한 산술에서는 같다.
입력 float32 변환 전에 큰 offset 때문에 작은 변화가 사라지면 복구하지 못한다. 표본 간격이
다르면 lag·차분의 실제 시간도 달라지며, spectral entropy와 다른 시간 통계는 관측 길이에
영향을 받는다. 표본 간격의 근거가 없어 초 단위를 추정하지 않는다. 원시 진폭을 쓰는 기저
모델의 점수·순위까지 단위 불변인 것은 아니므로 다른 현장으로 적용할 때 측정 단위와
표본 간격의 일치 여부를 별도로 확인한다.

무차원 특징끼리도 범위는 다르다. 차분 비율·ACF 등의 중복 여부를 학습 family에서 점검하고,
필요한 결측 대체·변환·스케일링도 학습 fold에만 맞춘다. 전체 126개 prefix에서 먼저
맞추지 않는다. 같은 family의 CSV와 모든 q는 같은 fold에 둔다. 전처리 누출은
[scikit-learn의 검증 지침](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)을 따른다.
이 조회 계약은 추천기의 입력 후보를 정하며 추천 성능이나 충분한 독립 표본 수를 보장하지 않는다.

실행·채점이 끝난 추천 학습용 결과는 아래처럼 연결한다. 실패·미채점 결과를 0점으로 채우지
않으며 model/config/seed/head는 비교할 후보·실행 단위를 식별한다.

```sql
SELECT i.*, c.model, r.config_id, r.seed, r.score_variant, r.vus_pr
FROM recommendation_inputs AS i
JOIN results AS r USING (prefix_feature_id)
JOIN model_configs AS c USING (config_id)
WHERE r.status = 'complete' AND r.primary_score = 1;
```

기존 `run_ratio_tuning` 기본 명령이 DB를 만들고 모델 실행 전에 126개 prefix를 저장한다.
trial 완료마다 실행 증거를 연결하고 채점 후 VUS를 적재한다. 최종 백업에서
`recommendation_inputs.csv`까지 내보내며 126행과 원표·산식 신원을 확인한 뒤 영수증을 닫는다.
`--prepare`는 준비만, `--execute-only`는 미채점 상태이며 전체 완료가 아니다. DB 내보내기가
실패하면 완료로 기록하지 않고, 같은 명령을 재개해 저장된 모델 결과와 채점 원표를 사용한다.
기본 명령과 산출물 위치는 [Lightning 실행 안내](lightning_studio.md)를 따른다.

### SQLite 원표와 표시 형식

새 실험당 실행 환경의 로컬 파일 경로에 SQLite DB 하나를 만들고 Python 표준
`sqlite3`를 사용한다. 별도 DB 서버·ORM·벡터 DB·일반화된 저장 프레임워크를 추가하지 않는다.
기존 budget·manifest·ledger·실행 이력을 통째로 복제하는 DB를 만들지 마라.

중심 원표는 `prefix_features`, `channel_features`, `model_configs`, `results`다.
prefix_features에는 파일/q 연결·N·n·d·입력 요약을, channel_features에는 prefix ID와 채널별
통계를, model_configs에는 기존 config ID와 전체 설정·target_use를 둔다.
results에는 prefix/config/seed/head별 상태·실제 VUS와 기존 실행·점수·채점 증거 참조를 둔다.
source·recipe·budget·schema 버전은 DB의 실험 신원에 묶고 상세 이력은 기존 산출물을 참조한다.
추가 표는 기존 제외/실행 기록의 SQL 연결과 무결성에 필요한 최소한만 만든다.

| 연결·제약 | 요구 |
| --- | --- |
| prefix_features | prefix_feature_id 유일. 같은 새 실험의 동일 CSV/q 중복 금지. 기본 논리 기록 126개를 확인한다. |
| channel_features | prefix_feature_id + channel_index 유일. 이름이 우연히 같은 서로 다른 센서를 합치지 않는다. |
| model_configs | 기존 config_id 유일. config_id를 파일/q/seed/head 때문에 다시 만들지 않는다. 전체 설정은 JSON 또는 기존 설정 참조로 보존한다. |
| results | prefix_feature_id + config_id + seed + score_variant 유일. FK를 켜고 관련 prefix/config가 없는 결과를 거절한다. |
| 재시도·공유 실행 | 같은 결과를 append해서 늘리지 않는다. 시도 이력은 기존 로그에 보존하고 같은 논리 결과를 갱신한다. 공유 trajectory/물리 실행은 기존 ID로 연결한다. |

키 열은 NOT NULL로 두고 점수 variant는 기존 계약의 정규값을 사용한다. NULL head 때문에
중복 검사를 통과하거나 실제 실행·채점이 끝나기 전에 complete로 표시하지 않는다.

엑셀 11열에는 score_variant가 없지만 TSPulse 등의 주 평가 head를 섞거나 버리면 안 된다.
DB 내부 결과 키에는 이를 반드시 포함한다. 기본 11열 조회·내보내기는 한 CSV와 하나의
score_variant를 고정하고 head는 파일/시트 이름에 표시한다. 여러 head를 한꺼번에 보는
기술 조회에는 score_variant를 명시한다. model/config_id에 head를 임의로 붙이지 마라.
raw·smoothed·보조 head는 기존 계약대로 보존하고 보조 head에 없는 VUS를 만들지 않는다.

아래는 DB에서 저장된 값을 연결하는 조회 예시다. 출력 때 통계 함수를 호출하지 않는다.

```sql
SELECT
    p.csv_file AS CSV_file,
    p.q_percent, p.training_boundary, p.observed_row, p.input_column,
    p.prefix_feature_id, c.model, r.config_id, r.seed, r.status, r.vus_pr
FROM results AS r
JOIN prefix_features AS p ON p.prefix_feature_id = r.prefix_feature_id
JOIN model_configs AS c ON c.config_id = r.config_id
WHERE p.csv_file = :csv_file AND r.score_variant = :score_variant
ORDER BY p.q_percent, c.model, r.config_id, r.seed;
```

DB 안의 원본 파일 식별은 파일명 단독이 아니라 기존 CSV ID·source SHA에 묶는다.
현재 18개 패널의 파일명도 유일성을 확인한다. 미래에 이름이 같은 파일을 임의로 합치지 않는다.

기본 전달물은 DB와 첨부의 네 묶음에 대응하는 UTF-8 BOM CSV다. 입력 특징·채널별 특징·
모델 설정은 공통 원표로 한 번 내보내고, 성능 결과는 CSV/head별로 내보낸다.
엑셀의 네 블록을 같은 고정 행 위치에 계속 늘려 서로 덮어쓰지 않는다.
XLSX가 필요하면 같은 네 묶음을 별도 시트로 옮기는 기존 출력 기능을 재사용한다.
매 trial마다 통합 XLSX나 전체 CSV를 다시 쓰지 않고 완료·인수 시 DB에서 내보낸다.
내보내기는 DB의 조회 사본이며 별도 편집·양방향 동기화 대상으로 만들지 않는다.

### 구조적 제외와 불완전 결과

정적 제외 단위는 CSV/q/model/config다. 해당 사유와 파생 window/patch/topk 제약은 기존
feasibility 원표에 남기고 SQL에서 연결해 조회한다. seed/head별 제외 행을 복제하지 않는다.
제외된 조합에는 실행 시도·VUS 결과·빈 점수 배열·checkpoint·dummy 파일을 만들지 않는다.
해당 CSV/q의 특징은 남기고 다른 가능한 config의 실행은 유지한다. 이 구조를 실제 예산과
실행기의 파일 목록 대조로 검증한다. PaAno patch·batch와 GDN topk·검증 창의 경계도 확인한다.

실제 실행하다 실패·중단한 경우는 구조적 제외와 구분해 기존 시도 로그에 남긴다.
실제 시도한 결과의 미완료 상태를 기록할 수는 있지만 VUS를 0으로 채우지 않는다.
성공한 seed만 평균내거나 일부 누락을 숨겨 우승 후보를 고르지 않는다.
성능 집계는 실제 완료·채점 조건을 만족한 원표와 봉인된 선택 규칙만 사용한다.

### 실행 중 증거와 비용

엑셀의 네 기본 표에 모든 운영 필드를 펼칠 필요는 없다. 기존 snapshot·metadata·점수·
checkpoint·scaler·학습 로그·시도 이력에 이미 있는 정보는 ID·경로·SHA로 참조한다.
모델이 실행된 동안에만 얻는 정보가 빠져 있으면 필요한 계측·저장만 보완한다.

- 실제 window/patch/context·batch·학습률·cap·완료/선택 epoch/update·반복 노출량,
  평가 세션 길이/범위, 실제 scaler 상태와 적용 범위, 입력 정규화와 점수 calibration을 보존한다.
- PCA fitted model, PaAno encoder·memory,
  GDN scaler·선택 state·calibration, 최종/선택 checkpoint·loss 계산 방식은 기존 저장을 재사용한다.
- TSPulse의 실제 입력 StandardScaler, 내부 head min/max/range·공식 교정값, 출력 최대값·
  나눗수·fitted MinMaxScaler 상태는 native_calibration에 남겨 완료 검사·DB·인수에 연결한다.
  학습 파일은 개별 저장 직후 참조를 이력에 기록하고 부분 저장도 인수한다.
- raw/smoothed·채널 보조 점수의 variant·shape·source 범위·alignment·label slice·boundary/lookahead,
  evaluator·ell_max·source/checkpoint·package·seed 신원과 파일 SHA/bytes를 연결한다.
- 준비·전처리·학습·test·calibration·저장/검사·전체 시도 시간과 측정 범위를 구분한다.
  새 경로에 없는 validation 단계를 별도 행 수·항상 0인 비용 열로 추가하지 않는다.
- 실제 CPU/GPU backend·환경, 적용 가능한 GPU allocated/reserved peak와 시작 기준값,
  CPU RSS 시작/종료와 확보 가능한 peak의 측정 범위, tracemalloc과 CUDA 측정 종류를 구분한다.
  프로세스 생애 peak를 현재 trial peak로 부르거나 GPU 0을 RAM 0으로 바꾸지 않는다.
- 새 checkpoint/scaler bytes와 사전학습 원본 bytes를 구분한다. TSPulse·target-free의
  공유 실행 비용을 q/config/head별로 중복 합산하지 않는다. 누적/추가 비용과 실제 실행량을 남긴다.
  측정 불가·강제 종료의 미확정 값은 NULL과 사유로 두고 추정해서 채우지 않는다.

정상 종료한 학습의 `training_log.loss_history`에는 PaAno의 iteration별 total·triplet·pretext
loss, pretext 가중치·학습률과 GDN의 epoch별 학습 batch MSE 합·평균·검증 loss를 남긴다.
PaAno의 `selected_iteration`과 GDN의 `selected_epoch`는 전체 완료량과 구분한다. 이력은
기존 학습 로그 JSON으로 저장하며 학습식·난수·최적 checkpoint 선택에는 사용하지 않는다.
실패한 학습의 부분 loss 곡선까지 복구하는 기능은 아니다.

feature 수집 시간·크기·저장 overhead는 모델 비용과 구분한다. 현장 단가·수집속도·미래 기간·
기업 계획 N은 실험에서 관측한 값처럼 만들지 않는다. 별도 자원 상한 실험·profiler는 추가하지 않는다.

### 저장·재개·인수

준비·봉인 → 126개 prefix/채널 특징 계산·검증 → 모델 실행과 즉시 저장 → 채점·선택 →
DB 연결·내보내기·전체 인수 순서로 연결한다. 원본 feature를 읽지 않는 prepare는 새 예산·
schema·예상 기록을 만든다. 입력 특징 누락·계산 예외·저장 실패가 있으면 모델 배치를 열지 않는다.
통계적 미정의 NULL은 이 실패에 포함하지 않는다.

DB 쓰기는 부모 프로세스 한 곳으로 모으고 트랜잭션을 사용한다. 파일 산출물과 SQLite는
하나의 트랜잭션이 아니므로, 실제 산출물을 원자적으로 저장·검증한 뒤 관련 DB 기록을 확정한다.
중간 실패를 전체 완료로 표시하지 않는다. 저장 실패 시 남은 배치를 멈추고 이미 얻은 결과를
보존한다. 같은 새 실행에서 누락된 DB 연결·내보내기만 복구하며 모델을 자동 재학습하지 않는다.
점수 원표와 DB 복사본은 기존 논리 키·score/evaluator 신원·값으로 대조한다.

`--execute-only`도 특징과 실행 증거를 남기며 finish-only/selection-only는 같은 새 실험의
결과를 소비하고 추천 자료 상태를 검사한다. DB 복구를 이유로 폐기한 옛 결과를 읽지 않는다.
실행 완료·채점/선택 완료·추천 자료 완료를 구분하고 세 조건을 충족해야 최종 완료다.

새 산출물은 기존 `experiments/01_ghl_main/` 안에 둔다. 계약은
`snapshots/dev18_selection/full_prefix_v2/`, SQLite DB·특징·연결·내보내기는
`results/dev18_tuning/full_prefix_v2/recommendation_evidence/`에서 관리한다.
경로·schema는 한 번 정의하고 .gitignore와 모든 소비 경로를 맞춘다.
기존 모델·원표 저장 경로를 이 작업 때문에 전면 재구성하지 마라.

최종 영수증에는 DB schema·실험 신원·특징·연결·내보내기의 지문과 예상/완료/누락 수를 묶는다.
DB가 닫혀 쓰기가 끝난 사본이나 SQLite backup으로 만든 일관된 사본을 전달·해시한다.
쓰는 중인 DB 본체만 복사하지 않는다. 재개 뒤에는 최종 지문을 다시 확정한다.
새 전달 묶음에는 DB·작은 원표·manifest·ledger·선택표·시도/비용 이력·필수 증거를 포함한다.
큰 점수·checkpoint·공식 가중치는 실제 보존 위치·SHA를 인수 목록에 남긴다.
단순 파일 존재나 헤더만 있는 CSV를 완료 증거로 쓰지 않는다.

### 구현 위치와 최소 검증

순수 feature 계산은 src/, Dev18 수집·SQLite 연결·내보내기는 tests/ghl_main/,
회귀 검사는 tests/unit/에 둔다. `run_ratio_tuning.py`의 준비·실행·종료/전달과
`run_dev18_tuning.py`의 지연 입력·등록 모델 실행·저장·채점·재개 호출을 함께 확인한다.
후보는 build_ratio_tuning_budget.py·model_feasibility.py, 실행 증거는
save_model_artifacts.py·execution_evidence.py·record_run_history.py를 재사용한다.
실제 변경한 저장 schema의 모든 writer/reader/검증을 맞추고 adapter 전체를 리팩터링하지 않는다.

아래 관련 회귀만 작성하고 실행은 허용된 원격 환경에 남긴다.

- 옛 튜닝 파일 없이 prepare가 성립하고 옛 baseline/점수 의존이 없다. q5를 포함해 전체 prefix와
  수정된 후보·scaler가 실제 호출에 적용되며 새 저장에 중복 fit/validation 행 수·별칭이 없다.
- 첨부 네 묶음의 열 순서·alias·숫자형·NULL·config/prefix 관계가 맞고, 여러 head가 11열
  출력에서 섞이지 않는다. config_id는 head 때문에 바뀌지 않는다.
- 동일 CSV/q의 여러 model/config/seed/head에서 extractor 호출은 한 번이다. 재개하면 저장된
  값을 읽고, q/범위/source/schema/계약 변경 때는 새로 계산한다. 정렬을 바꿔도 연결이 유지된다.
- 작은 입력으로 mean/std/median·상수 판정·ACF lag1·절대상관 중앙값과 유효 수/사유를 확인한다.
  짧음·상수·IQR 0 비상수·결측·비유한 값·유효 채널/쌍 없음, q별 통계 변화가 포함된다.
- prefix 뒤·평가 구간·라벨 변화는 현재 입력 통계에 영향을 주지 않는다. source 신원 변경과
  통계 수치 불변을 구분한다. 수집 유무로 입력/RNG/모델 인자·점수·선택 결과가 달라지지 않는다.
- 126개 논리 입력·채널 기록의 완전성, FK·UNIQUE, 재시도 시 결과 중복 방지, 원표/DB 값 일치를
  확인한다. 구조적 제외에는 seed/head별 dummy와 점수 파일이 없고 가능한 파일은 남는다.
- 공유 실행 비용 중복 방지, 실제 backend·메모리 측정 범위·미확정 값, 파일 저장/DB commit 사이
  중단·저장 실패·재개·일관된 DB 백업과 내보내기 복구를 확인한다.
- 전체 완료는 실행·채점/선택·추천 자료 인수를 모두 요구한다. 원격 전달 명령·예상량은 새
  budget을 사용하며 옛 Lightning/Colab의 고정 행수나 reset을 재사용하지 않는다.

완료 보고에는 첨부 형식의 반영, 제거한 중복 필드, 특징 재사용 단위, SQLite 연결·내보내기,
수정 파일·정적 검토·원격 검증 전 항목을 적는다. 입력 descriptor를 줄였음을 명시하고,
추천 성능 검증이나 본 튜닝을 이미 끝냈다고 보고하지 않는다. 다음 시작점은 관련 원격 회귀와
소규모 저장 경로 검증이며, 통과한 코드·계약·새 예산을 봉인한 뒤 별도 명령으로 전면 튜닝한다.
