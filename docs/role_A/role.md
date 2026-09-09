# A(강혁) — 데이터 EDA·Manifest

강혁님은 GHL·HAI 본실험 데이터와 TSB 튜닝 패널의 데이터 검수·manifest 승인을 맡습니다.
모델 구현, checkpoint 확인과 HPO는 강혁님의 책임이 아닙니다.

## 데이터 역할

공식 `TSB-AD-M-Tuning.csv` 20개는 provenance 목록입니다. 이 중 GHL 09·18을 제외한 비-GHL
18개만 사전 튜닝에 씁니다. GHL 09·18을 포함한 GHL25 전체가 주실험이고, HAI 23.05는
`train1 → test1`, `train1+train2 → test2` 두 실행의 외부 확인입니다.

`Dev18`이라는 이름은 별도 실험 데이터셋이 아닙니다. 기존 저장 경로의 내부 식별자일 뿐이며,
강혁님 문서에서는 “TSB 비-GHL 튜닝 패널 18개”라고 써 주세요.

## 확인할 항목

`manifest.md`에 파일별로 아래 정보를 확정합니다.

- 역할, 상대 경로, 파일명, 바이트 크기와 SHA-256
- 행 수, feature 수, feature 이름과 순서
- 정상 학습·테스트 경계와 prefix 기준 길이 `N`
- numeric 여부, NaN·Inf·결측 비율과 중복 timestamp
- 정상 학습 구간의 라벨 오염 여부
- constant·IQR 0·고상관 채널 현황
- 테스트 라벨 길이, 이상 구간 개수와 길이의 최소·중앙·최대
- 학습 구간만으로 계산한 채널별 자기상관 후보와 lag 상한
- HAI 파일별 독립 세션 여부와 86개 센서 순서

공식20과 TSB 튜닝 패널, GHL25를 별도 manifest 역할로 표시합니다. 튜닝 패널과 GHL25의 파일
교집합은 0건, 공식20과 GHL25의 교집합은 GHL 09·18 두 건이어야 합니다. 모든 `q`에서 튜닝
파일 목록은 같고 각 파일의 정상 학습 구간 앞쪽 prefix 길이만 달라집니다.

## EDA 해석 범위

constant·저분산·고상관은 데이터 특성으로 기록하지만 채널을 삭제하지 않습니다. 별도 저분산
threshold나 downsampling 배율도 제안하지 않습니다. 주기성은 지우님이 `ℓ_max`를 정할 때 보는
근거이며 모델 window를 대신 결정하지 않습니다.

테스트 라벨은 이상 구간 통계와 주혜님의 난이도 분석에만 씁니다. 모델 분할, scaler, HPO와
normalization 기준은 정상 학습 구간에서 정합니다.

## 인수물과 완료 신호

최종 인수물은 `docs/role_A/manifest.md`입니다. 데이터별 표와 판단을 남기고 조사 과정, 모델
구현 지시나 성능 예상은 적지 않습니다. 모델 담당자가 파일을 다시 읽지 않고 정적 feasibility를
계산할 수 있을 만큼 길이·feature·경계·품질 정보가 채워지면 완료입니다.

모델 source·checkpoint SHA, GDN 구현, TimeRCD·TSPulse forward는 모델
담당자에게 넘깁니다. 지우님의 VUS-PR·`ℓ_max` 구현과 주혜님의 통계도 대신 맡지 않습니다.
