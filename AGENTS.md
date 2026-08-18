# AGENTS.md — 작업 규약

이 파일은 이 저장소에서 작업하는 **모든 세션이 작업 시작 전에 반드시 읽어야 하는 규약**이다.
아래 규칙은 개별 세션의 편의보다 우선한다.

## 세션 시작

- `AGENTS.md` 다음에 `docs/lead/plan_v4.md`, `docs/lead/process_0_preverify.md`,
  `docs/lead/next_session.md`를 읽는다. 현재 결정을 바꿀 때만
  `docs/lead/decisions.md`를 찾는다.
- `docs/lead/next_session.md`의 현재 게이트를 확인하고, 직전 변경만 가볍게 검증한 뒤
  허용된 작업 하나를 수행한다. 완료 보고 전에는 다음 작업으로 넘어가지 않는다.
- 계획과 저장소 상태가 어긋나면 코드를 추측해 이어 쓰지 말고 계획 문서의 상태부터 고친다.

## 실험 분리

- 모델 본체와 모델 공통 실행 코드는 `src/`, 실험 설계·조합 실행·완료 판정·검증 코드는
  `tests/`, 실행 산출물만 `experiments/`에 둔다. `experiments/`에 Python 코드나 설명
  문서를 넣지 않는다.
- `tests/ghl_main/`은 `experiments/01_ghl_main/`, `tests/hai_extension/`은
  `experiments/02_hai_extension/`, `tests/checks/`는 `experiments/checks/`의 산출물을
  만든다. 순수 단위 검증은 `tests/unit/`에 둔다.
- 실험 산출물은 아래 세 루트 밖으로 나가지 않는다. 임의의 최상위 실험 폴더를 추가하지 않는다.
  - `experiments/01_ghl_main/`: GHL 세 계층 점수와 최종 교차점·곡선·난이도·비용 결과
  - `experiments/02_hai_extension/`: HAI 대표 점수와 인접행렬·그래프 안정성 결과
  - `experiments/checks/`: 데이터셋과 외부 참고 구현의 사전 점검 결과
- 모델 점수는 `scores/tier1|tier2|tier3/{model}/`, 배치 로그는 `logs/`, 실험 공통
  설정은 `snapshots/`, 최종 표·그림은 `results/{result_name}/`에 둔다. GDN처럼 한 실행의
  checkpoint·snapshot·점수를 함께 검사해야 하는 모델은 모델 폴더 아래 실행 단위로 묶는다.
- GHL 최종 결과 이름은 `crossover`, `three_tier_curves`, `difficulty_recovery`,
  `runtime_cost`다. HAI는 `baseline_curves`, `graph_stability`다.
- 새 데이터셋 감사는 `experiments/checks/datasets/{dataset}/`, 외부 코드 검증은
  `experiments/checks/reference_code/{implementation}/`에 둔다. 본 실험으로 승격할 때만
  `docs/lead/plan_v4.md`와 `docs/lead/decisions.md`를 고쳐 새 실험 루트를 만든다.
- 실험 간 파일 공유 금지. 공유가 필요하면 파일이 아니라 `src/`의 코드로만 한다.

## 일회성 점검

- 한 번 쓰고 버릴 점검 코드와 원시 결과는 저장소나 프로젝트 로컬 경로에 파일로 만들지 않는다.
  PowerShell·Python 인라인 명령으로 실행하고, 발견한 문제·결론·실제 조치만
  `docs/lead/process_0_preverify.md`나 `docs/lead/decisions.md`에 짧게 남긴다.
- 도구가 임시 파일을 꼭 요구하면 시스템 임시 폴더에 만들고 같은 작업 안에서 삭제한다.
- 데이터·버전·환경이 바뀔 때 같은 절차를 다시 실행해야 하는 감사와 dry-run만
  `tests/checks/`에 보존한다. 재실행 조건을 설명하지 못하는 점검 파일은 남기지 않는다.

## 파일명 규약

- 점수 배열 파일명:

  ```
  {dataset}__{series}__{model}__{tier}__r{ratio}__s{seed}__{raw|smoothed}__{trainnorm|testnorm}.npy
  ```

  예: `GHL__03__GDN__t2__r010__s1__raw__trainnorm.npy`

- 기본 산출물은 `trainnorm`(학습/검증 구간 추정 정규화), 부록 대조는
  `testnorm`(저자 원본 방식·테스트셋 추정).
- `ratio`는 세 자리로 쓴다: `005` / `010` / `020` / `050` / `100`.
- `series`는 두 자리 제로 패딩.
- 점수 배열은 항상 `(raw, smoothed)` 2벌을 저장한다.
- 1급 산출물은 집계 후 배열. 집계 전 채널별 점수 배열은 접미사 `__channels`를 붙인
  보조 산출물로 보존한다.
- smoothed는 자체 구현만 쓰며 GraGOD `smooth_scores`를 호출하지 않는다. 시간축 후행
  4-창, 처음 3개 시점 0, 정규화 → smoothing → 집계 순서다.
- GDN 1-step forecast 점수 길이는 `L-W`, 라벨은 `labels[W:]`다.

## 재현성

- 모든 실행은 학습 전에 config 전체, 입력 SHA-256, package 버전, 사용한 소스 버전을
  `snapshots/`에 JSON으로 저장한다. 작업 트리가 clean하지 않거나 설정에 고정한 GraGOD
  버전과 다르면 실행하지 않는다.
- 학습이 끝날 때 소스 버전과 clean 상태를 다시 검사한다. 실행 산출물 경로는
  `.gitignore`에 명시해 정상 산출물이 clean 판정을 깨지 않게 한다.
- 정규화 통계(median·IQR)는 **학습/검증 구간에서만** 추정한다. 테스트셋 추정 금지.
- 채널 집계는 `max`, 오차는 절대값.
- `docs/lead/process_0_preverify.md`의 실데이터 실행 전 조건이 모두 닫히기 전에는 GHL·HAI 모델을 학습하지 않는다.
  합성 dry-run과 단위 테스트만 허용한다.

## 역할 경계

- 저는 이 저장소에서 계층1·2·3의 점수·metadata·snapshot·시간 로그와 HAI GDN edge 집합까지만 만든다.
- VUS-PR·ℓ_max·threshold는 지우, 난이도·hit·Jaccard·Wilcoxon·TOST·bootstrap·교차점은 주혜가 맡는다.
- 외부 담당자의 코드를 이 저장소에 미리 구현하지 않는다. 산출물이 오면
  `docs/lead/process_0_preverify.md`에 지정된 파일과 `docs/lead/decisions.md`만 갱신한다.

## 판단과 기록

- 하이퍼파라미터와 구현 세부는 추측하지 않는다. 필요하면 다음 저장소의 실제 구현을 먼저
  확인한다:
  - `d-ailin/GDN` (main)
  - `GraGOD` (현재 고정 구현과 입력 축 두 줄 수정)
  - `TheDatumOrg/TSB-AD` (main)
- 조사 과정과 파일별 줄 번호는 문서나 코드 주석에 옮기지 않는다. 확인은 작업 전에 충분히
  하고 결과물에는 현재 결정과 필요한 이유만 짧게 남긴다.
- 코드 주석은 비자명한 동작과 제약을 설명할 때만 쓴다. 출처 목록, 결정 번호, 검토 이력은
  주석으로 남기지 않는다.
- 저장소마다 값이 다르면 `configs/`에 두고 `docs/lead/decisions.md`에는 최종 선택과 이유만
  자연스러운 문장으로 적는다.
- GraGOD의 post_process_scores 경로(정규화·smoothing)는 사용 금지, 후처리는 전부 우리
  코드가 소유한다.
- GraGOD 수정은 입력 축을 바로잡는 두 줄이 전부다. 추가 수정은 별도 검토 없이 하지 않는다.
- 수정 내용은 `src/models/tier2/gdn/gragod_input_transform.diff`에서 관리한다.

## 코드 스타일

- **ponytail 원칙으로 짠다**: 가장 게으르고 짧은 해가 정답이다. 추측성 추상화·미래 대비
  스캐폴딩 금지, stdlib·기존 코드 재사용 우선, 최소 diff. 단 이해를 건너뛰는 게으름은
  금지 — 문제를 다 읽고 나서 짧게 짠다. 의도적 단순화로 한계를 남길 때는
  `# ponytail:` 주석으로 한계와 업그레이드 경로를 남긴다.
- 함수·파일 이름은 동사+대상의 직관형으로 짓는다.
  예: `take_training_prefix`, `estimate_normalization_stats`, `save_score_arrays`.
- 약어 금지. 한 글자 변수 금지(루프 인덱스는 예외).
- 파일 하나 = 역할 하나. 유틸 잡동사니 파일(`utils.py`) 금지.

## 문서 작성 스타일

- 모든 한국어 산문(.md 보고서, 완료 보고, 주석의 서술문)은 설치된
  **humanize-korean 룰북을 작성 시점부터 적용**해 사람이 쓴 글처럼 쓴다.
- 핵심만 추리면: 번역투를 걷어낸다("~에 대해"는 목적격으로, 이중 피동 금지, "~를 통해"
  남발 금지, "가지고 있다"류 직역 금지). AI 관용구를 쓰지 않는다("결론적으로",
  "시사하는 바가 크다", "다음과 같은", hype 어휘). 문두 접속사·"~할 수 있다"·
  hedging을 남발하지 않고 단언한다. 종결어미와 문장 길이를 변주한다. 본문 볼드·
  이모지·대구 반복·불릿 남발을 자제한다. 고유명사·수치·인용은 한 글자도 건드리지 않는다.

## 진행 규율

- 0/1/2/3/4/5단계는 `docs/lead/plan_v4.md`의 프로젝트 단계만 가리킨다. 내부 구현 작업을 같은
  숫자로 다시 이름 붙이지 않는다.
- 현재 게이트의 작업을 마치면 다음 작업으로 스스로 넘어가지 않고 완료 보고를 한다.
- 완료 보고에는 현재 게이트, 바꾼 내용, 실행한 최소 검증과 결과, 다음 시작점을 적는다.
  조사한 파일과 줄 번호를 나열하지 않는다.

## 검증 비용 규율

- 검증은 변경 위험에 맞춘 최소 증거만 남긴다. 같은 근거를 확인하려고 동일한 고비용 검사를
  한 단계 안에서 반복하지 않는다.
- 직전 단계 검증은 항상 하되 산출물 존재, 설정 파싱, 로그 행 수, 기록된 판정값 대조로 끝낸다.
  직전 단계의 계산 코드를 바꾸지 않았다면 원본 데이터를 다시 읽지 않는다.
- Markdown만 바꿨으면 링크·경로·핵심 수치의 일치만 본다. YAML만 바꿨으면 파싱과 변경 키만
  검사한다. 순수 함수는 관련 단위 테스트만 실행한다.
- 전체 테스트는 관련 코드 묶음이 끝났을 때 한 번만 실행한다. 하위 작업에서 관련 테스트가
  통과했다면 문서 수정 뒤 전체 테스트를 되풀이하지 않는다.
- GHL·HAI 전체 CSV 스캔과 SHA-256 재계산은 최초 봉인, 데이터 버전·크기·수정 시각 변경,
  스키마·EDA·전처리 계산 코드 변경 때만 허용한다. 그 밖에는 기존 inventory와 snapshot을 쓴다.
- 지표 계산 코드가 바뀌면 영향을 받는 실험만 한 번 다시 계산한다. 다른 데이터셋이나 모델까지
  넓히지 않는다.
- 검증 명령이 열 이름·경로를 잘못 짚어 실패하면 산출물 오류와 구분한다. 헤더나 실제 경로를
  한 번 확인해 검증 명령만 고치며, 산출물 근거가 깨지지 않았다면 대형 계산을 다시 돌리지 않는다.
- 실제 모델 학습과 전체 배치는 `docs/lead/next_session.md`가 허용한 단계에서만 실행한다.
