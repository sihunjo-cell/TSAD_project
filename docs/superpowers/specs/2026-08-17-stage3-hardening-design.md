# 3단계 GDN 1차 보강 설계

## 목표

3a·3b의 데이터 경계와 GDN 실행 경로를 4단계 전에 다시 감사한다. 잘못된 상태를 보정해 실행하는 대신 학습 전에 멈추게 만든다. 저장공간은 이번 감사 대상에서 뺐다.

실제 GHL·HAI 학습과 합성 모델 학습은 수행하지 않는다. 작은 fixture 단위 테스트와 정적 검증만 쓴다.

## 확인한 결함

1. 고정 GraGOD `datasets/dataset.py:47-50,64-77`은 길이 L에서 `L-W`개 target을 만든다. `models/gdn/model.py:307-316`과 `models/predict.py:138-142`는 reconstruction 설명에 따라 마지막 값을 버려 GDN의 마지막 정상 forecast까지 잃는다.
2. Git 조회 실패가 `unknown` 문자열로 바뀌었고 snapshot이 학습 뒤에 생겼다. dirty 소스나 실행 중 변경을 막지 못했다.
3. batch 조합의 비율·seed가 YAML과 코드에 중복됐고 허용 비율도 분할기·파일명 코드와 갈라질 수 있었다.
4. loader는 label 유한성·이진성을 검사하지 않았다. `MinMaxScaler.transform`의 반환값을 버려 `copy=False` 동작에 기대고 있었다.
5. 입력 실물의 크기·SHA-256, 일부 package 버전, 저자 `pytorch-lightning` 포크 commit이 실행 snapshot에 없었다.
6. validation loader가 train의 shuffle 설정을 그대로 받았다.
7. 학습·추론 시간과 GHL −TOPK·TopK 민감도 실행 경로가 없었다.
8. 대조 팔과 back-trim 산출물이 주 실행과 같은 이름이나 폴더에 섞일 여지가 있었다.

## 설계

### 실행 봉인

- TSAD와 GraGOD의 Git 조회가 실패하거나 작업 트리가 dirty면 학습하지 않는다.
- GraGOD HEAD는 `485e26b0c6b1d63f4f3531c8d05597db82e9db29`와 정확히 같아야 한다.
- 입력 지문, runtime, 실행 config, 두 commit을 의존성 import와 학습보다 먼저 snapshot에 쓴다.
- 실행 끝에 두 HEAD와 clean 상태를 다시 검사한다.

### 점수와 입력

- predict 출력 전체를 이어 붙여 `X[W:]`와 직접 절대 오차를 계산한다. 점수 길이는 `L-W`, metadata의 라벨 범위는 `[W,null]`이다.
- 모든 세션은 같은 feature 수의 유한한 2차원 배열이며 길이가 `W+1` 이상이어야 한다.
- GHL·HAI label은 길이가 맞는 유한한 0·1만 허용한다.
- scaler의 반환 배열을 저장한다. train만 shuffle하고 validation은 shuffle하지 않는다.

### 설정과 데이터 원본

- 비율과 seed는 YAML에서 읽는다. 파일명·분할·조합 코드는 지원 비율 `5·10·20·50·100` 한 상수를 공유한다.
- GHL 25개, HAI 8개 CSV의 크기와 SHA-256을 `configs/input_manifest.yaml`에 둔다. batch 시작 때 한 번 검증하고 실제 실행에 해당하는 지문을 snapshot에 넘긴다.
- package 공개 버전을 모두 대조한다. `pytorch-lightning`은 `direct_url.json`의 설치 URL과 commit `834dbf3039ee82a2ac5e65eed25f9989222283c6`도 확인한다.
- 고정 포크가 숨겨 가진 MSE, horizon 1, Adam, scheduler, gradient clipping 계약을 YAML에 기록하고 실행 전에 대조한다. validation shuffle false는 포크값이 아니라 우리 러너의 고정 결정으로 구분한다.

### 산출물과 대조 팔

- `timing.json`은 학습, train-reference 추론, test 추론 시간을 따로 기록한다.
- GHL −TOPK는 25×2비율×3seed=150 fit이다.
- TopK 민감도는 `k={2,5,10}`의 450개 논리 조합이다. k=5의 150개는 주 실행을 재사용해 추가 fit은 300개다.
- 모델 식별자는 `GDN_NOTOPK`, `GDN_K2`, `GDN_K10`이다. back-trim과 대조 팔은 별도 상대경로를 보존한다.

## 완료 조건

- 바뀐 동작마다 기존 코드에서 실패하는 테스트를 먼저 확인한다.
- 고정 환경의 전체 단위 테스트, 실제 runtime 계약, 전체 YAML 파싱, compileall, `git diff --check`가 통과해야 한다.
- 이번 1차 보강에서는 commit과 합성 드라이런을 하지 않는다. 사용자 검토 뒤 commit한 clean HEAD에서 합성 드라이런을 통과해야 새 실행본이 봉인된다.
