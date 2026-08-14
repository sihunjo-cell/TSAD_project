# 결정 로그 — 레포 확인·규약 변경은 전부 여기에 한 줄씩

- 2026-08-14 exp00: GraGOD 레포는 github.com/GraGODs/GraGOD가 맞다. README.md:112가 d-ailin/GDN 기반임을 밝히고 있어서다. develop HEAD는 ec8cd452a410ba903a31beb097a010ba0448c095였고, 고정 해시 선택은 이때 보류했다(후보 3개는 RECON.md [A]).
- 2026-08-14 exp00: GDN 논문 값은 arXiv 2106.06947의 ar5iv HTML 렌더(§4.4, §3.6)에서 확인했다. PDF를 직접 파싱할 수 없는 환경이었다.
- 2026-08-14 exp00: 커밋이 "GDN 관련"인지는 두 기준으로 판정한다. models/gdn/** 을 직접 바꿨으면 ●, GDN이 함께 쓰는 datasets/·gragod/predictions/·gragod/training/·models/predict.py를 바꿨으면 ○.
- 2026-08-14 규약 변경: 코드 스타일에 ponytail 원칙을 추가했다(간결·직관·최소, 추측성 추상화 금지). 사용자가 직접 지시했다.
- 2026-08-14 규약 변경: 문서 작성 스타일 절을 새로 만들었다. 한국어 산문은 humanize-korean 룰북(github.com/epoko77-ai/im-not-ai, 로컬 클론 ../im-not-ai @53e24e8)을 쓰는 시점부터 적용한다. 이것도 사용자 직접 지시다.
- 2026-08-14 윤문 승인: 동결돼 있던 plan_v4.md·role_A/B/C.md·DECISIONS.md 결정 문구를 "의미 불변 + 문체만" 조건으로 humanize-korean 윤문했다. 사용자가 직접 승인했다(D-19의 재구성 금지·부탁문 원문 유지 지시에 대한 1회 예외이며, 수치·해시·절 번호·참조는 그대로다).

## 확정 결정 (2026-08-14, 결정 기록 세션)

### D-01
- 결정: GraGOD는 ec8cd452 커밋을 베이스로 잡고, 여기에 D-03 패치를 얹은 우리 포크의 커밋을 최종 고정 대상으로 삼는다. 패치 커밋 해시는 패치 세션을 마친 뒤 이 항목에 덧붙인다.
- 근거: RECON [A]. SHAPEFLOW 검증도 이 커밋 위에서 수행됐다.
- 기록일: 2026-08-14
- 패치 커밋 해시: 485e26b0c6b1d63f4f3531c8d05597db82e9db29 — 브랜치 fix/gdn-input-transform, 베이스 ec8cd452. 2026-08-14 패치 세션에서 기입했고, PATCH_REVERIFY.md의 재검증을 전 항목 통과했다.

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
- 결정: 인접행렬 추출은 RECON [F]의 후보 2로 간다. best.ckpt의 embedding.weight로 cos-sim → topk를 다시 계산한 "best validation 시점의 그래프"를 exp03 산출물로 삼고, 재계산 로직의 출처(model.py:137-144)를 주석으로 남긴다. Jaccard를 계산하기 전에 self-edge는 제거한다 — 모든 seed에 똑같이 존재하는 상수 edge라 일치도를 부풀리기 때문이다.
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
- 확정 버전(2026-08-14 패치 세션): torch-geometric==2.5.3. 전체 고정 환경은 configs/environment.yaml에 있다 — python 3.10.20, torch 2.2.2+cpu, numpy 1.26.4, pytorch-lightning은 저자 포크 커밋 834dbf30(설치 성공, 대체 없음), tensorboardX 2.6.2.2. 선정 근거는 PATCH_REVERIFY.md [1].

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
- 결정: (질문5 해소) HAI의 k는 규칙 max(5, ceil(0.25N))을 따르면 15~22 범위다. 정확한 값은 Manifest가 HAI 버전과 시계열 단위를 고정해 N이 정해지는 시점에 나온다. config에는 규칙과 범위, 그리고 "정확값은 Manifest 확정 후"라고 적는다.
- 근거: 계획서 7-4
- 기록일: 2026-08-14

### D-18
- 결정: (질문6 해소) Jaccard 비교는 시드 쌍별 전수 비교다. seed 10개가 만드는 45쌍 전부의 Jaccard(self-edge 제거 후 edge 집합 기준, D-08)를 계산해 중앙값·사분위·범위의 분포로 보고한다. 기준 그래프 하나를 정해 대조하는 방식은 기준 선택이 자의적이라 쓰지 않는다.
- 근거: 결정 기록 세션(2026-08-14)의 질문 목록 6번; D-08
- 기록일: 2026-08-14

### D-19
- 결정: (질문4 해소) 계획서 참조가 저장소 안에서 닫히도록 docs/plan_v4.md 를 근거 문서 위치로 지정한다. 이 파일의 내용은 사용자가 확정본을 직접 넣는다 — 세션이 계획서를 재구성하거나 요약해 채우지 않는다. 그 세션에서는 빈 파일과 "사용자가 확정본을 붙여넣을 자리"라는 한 줄 안내만 만들었다.
- 근거: 결정 기록 세션(2026-08-14)의 질문 목록 4번
- 기록일: 2026-08-14
