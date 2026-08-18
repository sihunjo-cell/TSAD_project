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

GHL은 25개 시계열과 5·10·20·50·100% 조건을 쓴다. 5%는 최저 데이터 조건, 10%는 저데이터
조건, 20%는 중간 조건, 50%는 고데이터 조건, 100%는 전체 기준이다. 이 이름은 곡선을 보기
위한 사전 격자일 뿐 관계 구조 회복이나 최적 비율을 뜻하지 않는다.
각 조건은 원 학습 구간의 앞쪽 누적분만 사용한다. 이후 시점의 데이터만 골라 과거의 저데이터
상태를 대신하는 통제군은 두지 않는다. 현시점부터 데이터가 얼마나 더 쌓여야 하는지를 재는
프로젝트 질문과 맞지 않기 때문이다.
비율별 데이터 감사도 과거 누적분을 나중의 전체 구간 통계와 대조하지 않는다. 각 시점까지
관측한 데이터만으로 표본 수·채널 변화·분산을 검사한다.

HAI는 23.05의 센서 86개를 사용한다. CSV 한 파일을 연속 세션 하나로 보고 파일 사이에는
윈도를 만들지 않는다. 훈련 4세션은 한 모델을 학습하는 데 함께 쓰고 테스트 2세션은 따로
채점한다. 비율은 10%와 100%다. GDN 이웃 수 22는 재현 기준 회의 전 잠정값이다.

다운샘플 배율은 1이다. timestamp와 label은 모델 입력에서 뺀다. GHL과 HAI가 제공하는 원래
학습 구간에 학습 비율을 바로 적용한다.

선택한 학습 구간의 마지막 10%를 validation으로 쓴다. scaler는 train 부분에만 fit하고 같은
scaler를 validation과 test에 적용한다. GHL은 시계열마다 scaler 하나, HAI는 훈련 4세션을
합쳐 scaler 하나를 쓴다.

## GDN 실행 규칙

GraGOD GDN의 입력 축 순서를 바로잡는 두 줄 수정은
`src/models/tier2/gdn/gragod_input_transform.diff`에서 관리한다. 데이터는 전처리한 텐서를
자체 러너로 직접 주입한다. GraGOD의 점수 후처리는 쓰지 않는다.

raw는 채널별 정규화 점수를 max로 집계한다. smoothed는 같은 점수에 시간축 후행 4칸 평균을
적용한 뒤 max로 집계하며 처음 3개 시점은 0이다. 학습·validation 오차에서 구한 median과
IQR을 테스트에 적용한 `trainnorm`이 기본 결과다. 테스트 자체 통계를 쓴 `testnorm`은 부록
대조용으로만 만든다. IQR epsilon은 0.01이다.

현재 config의 window 5, embedding 64, hidden 128, output layer 1개, heads 1, dropout 0.2,
negative slope 0.2, batch 32, 최대 50 epoch, Adam `lr=0.001`, `weight_decay=0`, `eps=1e-8`,
`betas=(0.9,0.99)`, early stopping patience 10, validation 0.1은 합성 검증용 잠정값이다.
재현 기준·train stride·topk를 본 실험 전에 확정하고 config를 함께 고친다. GHL seed는 1~3,
HAI seed는 1~10이다.

1-step forecast 점수 길이는 `L-W`, 대응 라벨은 `labels[W:]`다. 여러 세션은 세션별 window
dataset만 합치며 원시 배열을 이어 붙이지 않는다. HAI 테스트 세션은 같은 학습 모델로 각각
점수를 만든다.

HAI 그래프는 best checkpoint의 embedding으로 다시 계산한다. self-edge 포함본과 제거본을
모두 저장하고 Jaccard에는 제거본을 쓴다. 이웃 비율 0.25는 데이터에서 최적화한 값이 아니라
현재 잠정 비교 규칙이며 데이터에서 고른 최적값은 아니다.

## 실행과 재현성

GHL GDN 본 실험은 `25×5×3=375`, HAI GDN 확장은 `2×10=20`개 학습 조합이다. GHL
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
