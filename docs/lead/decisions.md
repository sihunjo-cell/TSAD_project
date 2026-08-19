# 현재 결정

이 문서는 지금 유효한 결정만 적는다. 조사 과정, 파일별 줄 번호, 폐기한 후보는 남기지 않는다.
판단을 바꿀 때는 충분히 확인한 뒤 해당 문장과 관련 설정을 함께 고친다.

## 연구 범위와 역할

`plan_v4.md`가 연구 설계 원본이며 현재 준비 상태는 `process_0_preverify.md`가 관리한다.
저는 계층 1·2·3 모델과 공통 점수 생산을 맡는다. 강혁은 데이터와 Manifest, 주기성을
확인한다. 지우는 VUS-PR, threshold 250개, ℓ_max와 채점기 검증을 담당한다.
주혜는 난이도 분할, 구간 hit, Jaccard, 통계와 결과 해석을 맡는다.

외부 담당자의 결과가 오기 전에는 그 역할을 대신 구현하지 않는다. 실데이터 학습도 실행 전
조건이 모두 정리될 때까지 시작하지 않는다.

## 데이터와 전처리

GHL과 HAI는 5·10·20·40·60·80·100% 조건을 쓴다. 이 값은 저데이터 구간을 촘촘히 보고
40% 이후 증가 양상도 같은 간격으로 비교하는 사전 격자다. 관계 구조 회복 지점이나 최적
비율을 뜻하지 않는다.
각 조건은 고정 validation을 뺀 fit pool의 앞쪽 누적분만 사용한다. 이후 시점의 데이터만 골라 과거의 저데이터
상태를 대신하는 통제군은 두지 않는다. 현시점부터 데이터가 얼마나 더 쌓여야 하는지를 재는
프로젝트 질문과 맞지 않기 때문이다.
비율별 데이터 감사도 과거 누적분을 나중의 전체 구간 통계와 대조하지 않는다. 각 시점까지
관측한 데이터만으로 표본 수·채널 변화·분산을 검사한다.

HAI는 23.05의 센서 86개를 사용한다. CSV 한 파일을 연속 세션 하나로 보고 파일 사이에는
윈도를 만들지 않는다. 훈련 4세션은 한 모델을 학습하는 데 함께 쓰고 테스트 2세션은 따로
채점한다. GDN 이웃 수는 본실험 전에 정한 프로젝트 전이 규칙에 따라 GHL 5, HAI 22로 고정한다.

다운샘플 배율은 1이다. timestamp와 label은 모델 입력에서 뺀다. GHL과 HAI가 제공하는 정상
학습 구간에서 고정 validation을 먼저 분리한 뒤 남은 fit pool에 학습 비율을 적용한다.

전체 정상 학습 구간의 마지막 10%를 고정 validation으로 한 번만 분리한다. 나머지 fit pool의
앞쪽 5·10·20·40·60·80·100%를 시간순 누적 subset으로 쓴다. scaler는 현재 비율의 fit
subset에만 fit하고 같은 고정 validation과 test에는 transform만 적용한다. GHL은 시계열마다,
HAI는 같은 비율의 훈련 4세션 fit subset을 합쳐 scaler 하나를 맞춘다.

## 계층 2 모델 입력 길이

모델 입력 길이는 GHL의 모든 학습 비율에서 바꾸지 않는다. CI-AE와 LSTM-AD는 100, USAD는
10, GDN은 5다. HAI GDN도 5를 쓴다. CI-AE와 LSTM-AD는 채택한 TSB-AD wrapper의 실제
기본값을 따른다. USAD 10은 논문의 window 민감도 결과와 WADI 설정을 바탕으로 GHL 결과를
보기 전에 정한 transfer 기준이다. GDN 5는 논문의 SWaT·WADI 실험값이다.

이 값은 ACF나 테스트 성능으로 고른 최적값이 아니다. EDA는 고정한 입력 길이로 각 비율에서
학습·validation window가 생기는지만 검사한다. `vus_l_max`는 모델 입력 길이와 별개이며
지우가 맡는다.

CI-AE는 센서마다 AE 하나를 따로 학습하고 가중치를 공유하지 않는다. 그래야 채널 사이의
정보 전달을 막는 대조군이라는 정의가 유지된다.

학습 비율만 바꾼다. 모델 구조, 입력 길이, hidden·latent 차원, learning rate, batch, 최대
epoch와 early stopping, optimizer, 이상 점수, 전처리와 정규화는 모델별로 한 번 정한 뒤 모든
비율과 seed에 고정한다. 값이 충돌하면 논문의 데이터셋별 실험값, 공식 config·notebook,
공식 CLI 기본값, TSB-AD 기본값, 프로젝트 임의값 순으로 판단한다. GHL에 직접 대응하는 공식
값이 없는 USAD batch는 128, GDN topk는 GHL 5·HAI 22를 본실험 전 프로젝트 전이 규칙으로
고정했다. 성능을 본 뒤 바꾸지 않는다.

USAD는 논문의 WADI 항목에서 window 10, latent 100, 최대 70 epoch만 옮긴다. GHL 원 관측
간격은 유지하므로 WADI 전처리 전체를 복제한 profile이라고 부르지 않는다. 두 개의 Adam
`lr=0.001`, 조기 종료 없음, 점수 가중치 `(0.5,0.5)`도 고정한다. 공식 SWaT notebook의
batch 7,919는 SWaT window 수에 맞춘 값이라 GHL로 곧바로 옮길 근거가 없다. 따라서 batch
128을 프로젝트 고정 전이값으로 쓰고 모든 비율과 seed에 유지한다. 공식 학습은
batch마다 optimizer를 두 번 갱신하므로 update 수도 `2×batch 수×70`으로 계산한다.

학습 손실과 저장 점수는 구분한다. CI-AE·LSTM-AD·USAD·GDN의 학습 손실은 각 출처 구현의
MSE 규칙을 유지한다. 저장 점수는 공통 비교를 위해 채널별 절대오차를 쓴다. CI-AE와 USAD는
window 안의 절대오차를 채널별 평균하고 중앙 시점에 놓는다. 학습·validation 오차로 채널별
median·IQR 정규화를 맞춘 뒤 max로 집계한다.

재구성 모델은 원 wrapper의 복제 padding을 쓰지 않는다. CI-AE의 core 점수는
`source[50:L-49]`, USAD는 `source[5:L-4]`에 대응한다. LSTM-AD는 `source[100:L]`, GDN은
`source[5:L]`에 맞춘다. 실제 범위는 점수 metadata에 저장한다.

## GDN 실행 규칙

GraGOD GDN의 입력 축 순서를 바로잡는 두 줄 수정은
`src/models/tier2/GDN/model.py`에 보존한다. 데이터는 전처리한 텐서를
자체 러너로 직접 주입한다. GraGOD의 점수 후처리는 쓰지 않는다.

raw는 채널별 정규화 점수를 max로 집계한다. smoothed는 같은 점수에 시간축 후행 4칸 평균을
적용한 뒤 max로 집계하며 처음 3개 시점은 0이다. 학습·validation 오차에서 구한 median과
IQR을 테스트에 적용한 `trainnorm`이 기본 결과다. 테스트 자체 통계를 쓴 `testnorm`은 부록
대조용으로만 만든다. IQR epsilon은 0.01이다.

window 5, 최대 50 epoch, patience 10, betas `(0.9,0.99)`는 논문의 SWaT·WADI 설정을
따른다. 새 데이터셋에 바로 적용할 공통값은 공식 `run.sh`의 embedding 64, hidden 128,
output layer 1개, batch 32, stride 1, Adam `lr=0.001`, `weight_decay=0`을 쓴다. dropout 0.2도
원 저자 모델을 따른다. `run.sh`와 `TimeDataset`의 실제 동작에 맞춰 학습·평가 stride는 모두
1이다. 원 저자 학습 경로에는 scheduler와 gradient
clipping이 없지만 현재 GraGOD trainer는 둘을 강제한다. GraGOD는 dropout 0.2를 attention과
출력에 함께 적용해 원 저자 경로와도 다르다. 출처 선택은 끝났고 이 세 동작의 adapter만
남았다. validation 0.1은 프로젝트 공통 고정 분할 규칙이다. topk는 GHL 5·HAI 22로
고정했다. GHL
seed는 1~3, HAI seed는 1~10이다.

1-step forecast 점수 길이는 `L-W`, 대응 라벨은 `labels[W:]`다. 여러 세션은 세션별 window
dataset만 합치며 원시 배열을 이어 붙이지 않는다. HAI 테스트 세션은 같은 학습 모델로 각각
점수를 만든다.

HAI 그래프는 best checkpoint의 embedding으로 다시 계산한다. self-edge 포함본과 제거본을
모두 저장하고 Jaccard에는 제거본을 쓴다. 이웃 비율 0.25를 바탕으로 한 GHL 5·HAI 22는
본실험 전에 고정한 프로젝트 전이 규칙이며 데이터에서 고른 최적값은 아니다.

## 실행과 재현성

GHL GDN 본 실험은 `25×7×3=525`, HAI GDN 확장은 `7×10=70`개 학습 조합이다. GHL
실행에는 앞쪽 누적분만 쓴다.

실행 전에는 입력 크기와 SHA-256, 설정, package와 소스 버전, clean 상태를 확인한다.
snapshot은 학습보다 먼저 저장한다. 학습·train-reference 추론·test 추론 시간과 accelerator는
따로 기록한다. 점수, metadata, best checkpoint, early stopping 로그, snapshot과 필요한 HAI
edge가 모두 있고 종료 검증을 통과해야 `COMPLETE`로 인정한다.

실행 코드가 바뀌면 clean 환경에서 합성 dry-run을 한 번 통과시킨다. forecast 오차가 비유한
값이거나 embedding의 norm이 0이면 해당 산출물을 만들지 않는다. 입력 파일이나 loader가
바뀌지 않았다면 같은 대형 지문 검사를 반복하지 않는다.

## 폴더와 기록

모델과 공통 코드는 `src/`, 실험 실행과 검증 코드는 `tests/`, 실행 결과는 `experiments/`에
둔다. GHL 결과는 `experiments/01_ghl_main/`, HAI 결과는
`experiments/02_hai_extension/`, 반복 가능한 점검 결과는 `experiments/checks/`에 모은다.

한 번 쓰는 진단 코드는 저장하지 않는다. 필요한 경우 인라인이나 임시 폴더에서 실행한 뒤
바로 지우고, 프로젝트 문서에는 확인된 문제와 최종 조치만 짧게 남긴다. 코드 주석은 동작의
이유를 설명할 때만 쓰며 조사한 저장소와 줄 번호를 나열하지 않는다.
