# B(지우) — 채점기·ℓ_max

지우님은 모델을 구현하거나 실행하지 않습니다. 제가 계층 1·2·3에서 만든 점수 배열을 같은
규칙으로 채점하는 독립 평가자입니다.

## 입력 계약

점수 파일명은 아래 형식을 따릅니다.

```
{dataset}__{series:02d}__{model}__{tier}__r{ratio:03d}__s{seed}__{raw|smoothed}__{trainnorm|testnorm}(__channels).npy
```

`trainnorm` 집계본이 본 채점 대상입니다. `testnorm`은 부록에 따로 싣고 `__channels`가 붙은
배열은 회수 분석용으로만 씁니다. metadata의 `score_length`와 `label_slice`가 실제 배열과
맞지 않으면 채점을 멈춥니다. GDN 점수 길이는 `L-W`, 라벨은 `labels[W:]`입니다.

모델별 예외를 채점 코드에 하드코딩하지 않습니다. 모든 모델은 `score_interface.md`의 같은
점수·metadata 계약으로 받습니다.

## 구현할 것

VUS-PR을 주지표로 계산하고 점수 분위수 위치 250개를 threshold로 씁니다. AUPRC와
point-adjust를 적용하지 않은 F1은 보조 지표로 같은 원표에 둡니다. 테스트셋에서 최적
threshold를 찾는 경로는 본 결과에 쓰지 않습니다.

ℓ_max는 아래 세 후보를 비교한 뒤 채널별 자기상관 국소 최대의 중앙값으로 정합니다.

1. 채널별 자기상관 국소 최대의 중앙값
2. 이상 구간 길이의 중앙값
3. 공정 지식으로 정한 주기

이상 구간 길이는 비교와 탈락 사유에만 쓰고 선택값 최적화에는 넣지 않습니다. 선택값과 lag
상한은 실험 전에 고정합니다.

GHL의 `ℓ_max`는 332 samples로 확정했습니다. 전체 GHL 학습 prefix에서 timestamp와 label을
제외한 비상수 475개 채널의 정규화 ACF를 계산하고, lag 1--500의 첫 strict local peak가 검출된
125개 채널 lag의 중앙값입니다. 파일명의 `tr_` 경계 이전 행만 ACF에 사용하며, GHL의 모든
모델·비율·seed에 같습니다. 이전 345와 test 구간을 섞어 계산한 336은 사용하지 않습니다.
`ℓ_max`는 총 buffer window 폭이므로 TSB-AD 호환 VUS-PR에서는 각 이상 구간 양쪽에
`floor(ℓ / 2)`칸의 sqrt 감쇠 buffer를 둡니다. HAI의 `ℓ_max`는 165 samples입니다. 첫 시간
조건에서 사용 가능한 train1만으로 timestamp를 제외하고 같은 lag 1--500의 첫 strict ACF peak를
구했으며, peak가 검출된 42개 채널 lag의 중앙값입니다. 이 값은 test1과 test2 조건 모두에
동일하게 적용하므로 미래 train 세션을 사용하지 않으면서 평가 자도 바뀌지 않습니다.
계산 조건과 집계 결과는 `lmax_evidence.json`에 둡니다.

채점기는 무작위 점수가 낮은 값을 받는지, point-adjust가 점수를 부풀리는지 검증해야 합니다.
parser, 라벨 정렬, 상수 점수와 비유한 값 입력도 최소 테스트로 확인합니다. `ℓ/2`, `ℓ`,
`2ℓ` 비교는 본 규칙을 바꾸지 않는 부록 민감도 분석입니다.

합성 안전성 점검에서 무작위 score 20회의 평균 VUS-PR은 0.323으로 perfect score의 1.000보다
낮았고, point-adjust는 한 시점만 맞힌 긴 이상 구간의 F1을 0.095에서 1.000으로 부풀렸습니다.
따라서 본 결과에는 point-adjust를 쓰지 않습니다. 재현 결과는
`experiments/checks/scoring/safeguard_summary.json`에 둡니다.

출력은 `데이터셋 × 시계열 × 모델 × 비율 × seed × raw/smoothed × 정규화 방식`의 채점
원표입니다. 성능 해석과 계층별 우열은 쓰지 않고 주혜에게 넘깁니다.

점수 파일이 준비되면 아래처럼 본 채점 원표를 만듭니다. `scores-dir`에는 `trainnorm` 집계본과
각 집계본의 metadata가 있어야 하며, `dataset-dir`에는 해당 데이터셋의 원본 label 파일이 있어야
합니다.

```powershell
python -m src.채점기.score_runner --scores-dir <점수_폴더> --dataset-dir <원본_데이터_폴더> --output-csv <결과.csv>
```

HAI는 condition 01·02를 각각 원표 행으로 먼저 저장한다. 필요할 때만 `--hai-summary-csv`로
두 시간 조건의 비가중 산술평균 요약을 별도 파일에 만든다.

완료 신호는 파서 호환 확인, ℓ_max와 threshold 확정, 무작위·point-adjust 검증과 채점기
테스트 통과입니다.
