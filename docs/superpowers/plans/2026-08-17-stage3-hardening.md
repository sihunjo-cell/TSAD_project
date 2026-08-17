# 3단계 GDN 1차 보강 실행계획

설계 원본은 `docs/superpowers/specs/2026-08-17-stage3-hardening-design.md`다. 실제 모델은 돌리지 않는다.

## 1. 실행 상태 봉인

- [x] Git 조회 실패, dirty 작업 트리, 잘못된 GraGOD hash를 재현하는 테스트 작성
- [x] 학습 전 snapshot과 실행 후 HEAD·clean 재검사 구현
- [x] package 버전과 `pytorch-lightning` 설치 원본 commit 검증
- [x] 관련 테스트 RED→GREEN 확인

산출물: `src/common/verify_run_context.py`, `tests/test_verify_run_context.py`, runner snapshot 경계.

## 2. forecast 정렬 수정

- [x] 마지막 1-step forecast가 빠지는 실패 테스트 작성
- [x] 점수 길이를 `L-W`, 라벨 범위를 `[W:]`로 수정
- [x] GraGOD `datasets/dataset.py:47-50,64-77`, `models/gdn/model.py:292-316`, `models/predict.py:121-145` 대조
- [x] 인터페이스 문서와 과거 결정의 대체 상태 기록

산출물: `src/gdn_runner/run_gdn_single.py`, `docs/score_interface.md`, D-34.

## 3. loader 경계 보강

- [x] 비이진·비유한 label, 짧은 세션, 비유한 feature 실패 테스트 작성
- [x] scaler 반환 배열 저장
- [x] validation shuffle false 고정
- [x] GHL·HAI 세션 경계 유지 확인

산출물: `src/data_split/validate_labels.py`, 두 loader, `tests/test_load_gdn_inputs.py`.

## 4. 입력·설정 원본 고정

- [x] GHL 25개와 HAI 8개 파일 크기·SHA-256 manifest 작성
- [x] batch 시작 때 입력을 한 번 검증하고 snapshot에 파일 지문 전달
- [x] 실데이터 러너의 누락·잘못된 파일 지문 거부
- [x] 비율·seed를 YAML에서 읽고 지원 비율 상수를 한 곳으로 통합
- [x] 고정 포크의 숨은 학습 계약을 YAML과 실행 검증에 연결

산출물: `configs/input_manifest.yaml`, `src/common/verify_input_files.py`, `src/common/experiment_config.py`, D-36.

## 5. 산출물과 대조 팔 연결

- [x] `timing.json` 저장과 완전성 요구
- [x] GHL −TOPK 150 fit 명세와 별도 경로 연결
- [x] TopK 민감도 450 논리 조합, k=5 재사용, 추가 300 fit 연결
- [x] 모델 식별자와 back-trim 상대경로 충돌 방지
- [x] −TOPK snapshot의 사용되지 않는 topk도 YAML 기준값 유지

산출물: exp02 `check_controls.py`, `run_controls.py`, 두 batch 완전성 검사, D-37.

## 6. 문서와 정적 검증

- [x] `AGENTS.md`, `DECISIONS.md`, 현재 점검표와 다음 세션 계획 갱신
- [x] 과거 D-23·D-27의 점수 정렬이 현행 계약이 아님을 표시
- [x] 고정 환경 전체 unittest: 91건, 실패·오류 0건
- [x] 실제 runtime·설치 원본 계약 확인
- [x] 전체 YAML 5개 파싱과 조합 수 대조
- [x] Python compileall
- [x] `git diff --check`

최종 검증값은 `docs/pre_run_checklist.md`와 이번 완료 보고에 적는다. 실패가 하나라도 나오면 이 계획을 완료로 바꾸지 않는다.

## 사용자 검토 뒤 할 일

- [ ] 변경을 한 commit으로 봉인
- [ ] TSAD·GraGOD clean 상태 확인
- [ ] 합성 드라이런 한 번 실행: test 200·W 8 → 점수 192, `label_slice=[8,null]`
- [ ] snapshot·timing·점수 8개·metadata·checkpoint·early stopping·TopK 대조

이 네 항목은 이번 1차 보강 범위가 아니다. 끝나기 전에는 4단계로 넘어가지 않는다.
