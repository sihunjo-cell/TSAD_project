# 다음 세션 작업 기준

마지막 갱신일은 2026-08-17이다. 새 세션은 `AGENTS.md`, 이 문서, `DECISIONS.md` 순서로 읽는다.

## 현재 상태

3a의 데이터 정찰·Manifest·비율 EDA와 3b의 GDN 입력·러너·배치 구현은 끝났다. 4단계 전 1·2차 재감사에서 점수 정렬, 실행 봉인, 파일 지문, 환경 원본, 라벨·scaler, YAML 단일 원본, 시간 기록, GHL 대조 팔, 최종 성공 표식을 보강했다. 3차 감사는 이전 commit 산출물의 완료 오인과 HAI Jaccard의 혼합 실행 위험을 닫고 TopK self-edge 설명을 바로잡았다. 4차 감사는 preflight에 남은 `L-W-1` 표본 수를 `L-W`로 고치고 시간·Jaccard 산출물에 실행 신원을 붙였다. 5차 감사는 같은 확정값을 순환 수정한 기록이 없음을 확인하고, manifest 검증 뒤 실제 로드까지 외부 입력이 바뀌는 경계를 닫았다. 6차 감사는 실행 API와 재개 경로를 다시 추적해 새 코드 결함이 없음을 확인하고, 계획서에 남은 HAI 채널 범위만 확정값 86으로 고쳤다. 7차 감사는 패치 근거 경로, Jaccard 산출물의 Git 제외, GHL preflight의 현재 상태, GHL 25개 통계의 일반화 범위를 바로잡았다. 세부 결정은 D-33~D-44, 점검표는 `docs/pre_run_checklist.md`에 있다.

이번 변경은 아직 commit하지 않았고 새 합성 드라이런도 돌리지 않았다. D-31의 과거 봉인은 이전 구현에만 유효하다. 현재 코드를 실행 기준으로 봉인하려면 사용자 검토 뒤 새 commit과 clean 상태의 합성 드라이런이 필요하다.

| 단계 | 상태 | 근거 |
| --- | --- | --- |
| 3a | 완료 | `docs/manifest_draft.md`, `experiments/exp01b_ghl_preflight/`, `experiments/exp01c_hai_preflight/`, D-17·D-20·D-21 |
| 3b 기본 구현 | 완료 | GHL·HAI loader, GDN runner, exp02·exp03 batch, D-22~D-32 |
| 3b 1차 논리 보강 | 완료. 고정 환경 단위 테스트 91건과 정적 검증 통과 | D-33~D-38, `docs/pre_run_checklist.md` |
| 3b 2차 논리 보강 | 완료. EDA 봉인 로그 재계산, 고정 환경 단위 테스트 96건과 정적 검증 통과 | D-39, `docs/superpowers/plans/2026-08-17-stage3-second-audit.md` |
| 3b 3차 논리 보강 | 완료. 현재 commit과 snapshot 신원 결합, TopK 설명 정정, 고정 환경 단위 테스트 99건 통과 | D-40, `docs/superpowers/plans/2026-08-17-stage3-third-audit.md` |
| 3b 4차 논리 보강 | 완료. feasibility 표본 수 정렬, timing·Jaccard 신원 보강, 고정 환경 단위 테스트 100건 통과 | D-41, `docs/superpowers/plans/2026-08-17-stage3-fourth-audit.md` |
| 3b 5차 논리 보강 | 완료. 검증 뒤 입력 변경 차단, 순환 여부 판정과 감사 종료 기준 고정, 고정 환경 단위 테스트 104건 통과 | D-42, `docs/superpowers/plans/2026-08-17-stage3-fifth-audit.md` |
| 3b 6차 논리 감사 | 완료. 실행·재개 경로에 새 결함 없음, HAI 확정 채널 수 문서 정합화, 고정 환경 단위 테스트 104건 통과 | D-43, `docs/superpowers/plans/2026-08-17-stage3-sixth-audit.md` |
| 3b 7차 최종 논리 감사 | 완료. 재현 경로·Jaccard Git 제외·preflight 현재성·통계 일반화 범위 보강, 관련 단위 테스트 7건 통과 | D-44, `docs/superpowers/plans/2026-08-17-stage3-seventh-audit.md` |
| 새 실행본 봉인 | 실행 승인 | 이 변경을 commit한 뒤 clean 합성 드라이런으로 판정한다. |
| 4 | 스모크 1건 승인 | `GHL series 01·10%·seed 1`의 `COMPLETE`와 현재 commit snapshot이 완료 신호다. 전체 배치는 금지한다. |

## 현재 고정 계약

- GHL은 19채널·25시계열, HAI 23.05는 86채널·train 4세션·test 2세션이다.
- 주 비율은 GHL `5·10·20·50·100%`, HAI `10·100%`다. GHL back-trim은 `5·20·100%`다.
- GDN topk는 GHL 5, HAI 22다. window 5, batch 32, 최대 50 epoch, validation 0.1은 모든 비율에 같다.
- 점수 길이는 `L-W`, 라벨은 `[W:]`다. trainnorm 통계는 train·validation 오차에서만 추정한다.
- GHL·HAI feasibility 로그의 정규화 표본 수도 같은 `L-W`다. 실제 길이에서 validation 10% 경계의 반올림 불일치는 없다.
- 실행 전 두 저장소가 clean이어야 하며 GraGOD HEAD는 `485e26b0c6b1d63f4f3531c8d05597db82e9db29`여야 한다.
- 입력 크기·SHA-256·검증 시점 `mtime_ns`, package 버전, `pytorch-lightning` 설치 commit, 실행 config를 학습 전 snapshot에 남긴다. 실제 로드 직전·직후 파일 크기와 수정 시각이 달라지면 모델을 실행하지 않는다.
- 비유한 forecast 오차와 정의되지 않은 cosine graph는 저장하지 않는다. 배치는 종료 검증까지 끝난 뒤 `COMPLETE` 표식을 쓰며, 현재 TSAD·GraGOD hash와 snapshot의 두 hash가 같을 때만 재개 과정에서 완료로 센다.
- `timing.json`은 실제 accelerator를 기록한다. HAI Jaccard CSV는 각 행에 TSAD·GraGOD commit을 기록한다.
- Jaccard의 `analysis/`는 재생성 가능한 산출물로 Git에서 제외한다. Wilcoxon·TOST·bootstrap은 GHL 25개 task 내부 요약이며 모집단 추론에 쓰지 않는다.
- GHL −TOPK 추가 fit은 150개다. TopK 민감도는 450개 논리 조합 가운데 k=5 주 실행 150개를 재사용해 300개를 추가한다.

## 다음 시작점

7차 최종 논리 감사까지 끝났고 사용자가 새 실행본 봉인과 4단계 스모크를 승인했다. D-34 이후 점수 길이 교정은 한 번뿐이고 비율·seed·validation·window·topk는 번복되지 않았다. 새 diff·manifest 불일치·고정 외부 레포 변경·실패 테스트·dryrun 실패가 없으면 같은 정적 감사를 반복해도 읽기 전용 대조만 하고 코드를 고치지 않는다. 같은 대형 CSV를 다시 읽거나 SHA-256을 다시 계산하지 않는다.

아래 순서만 수행한다.

1. 전체 단위 테스트, 환경 계약, YAML 파싱, compileall, `git diff --check`를 한 번 확인한다.
2. 원본 데이터와 모델 산출물이 diff에 없는지 확인하고 변경을 한 commit으로 묶는다.
3. TSAD와 GraGOD 작업 트리가 clean인지 확인한다.
4. 합성 드라이런을 한 번 실행한다. test 200·W 8에서 점수 192와 `label_slice=[8,null]`을 확인한다.
5. snapshot의 두 commit, 입력·환경 계약, timing, 점수 8개, metadata, checkpoint, early stopping 로그, TopK 복원을 대조한다.
6. 1~5가 통과하면 `GHL series 01·10%·seed 1` 한 건만 실행하고 `COMPLETE`와 현재 commit snapshot을 확인한 뒤 멈춘다.

스모크 확인 전 전체 배치로 넘어가지 않는다. 새 세션은 `experiments/exp02_gdn_ghl/runs/series_01/r010/s1/`의 `COMPLETE`와 snapshot을 먼저 확인해 현재 완료 상태를 판정한다.
