# Dev18 Ratio-Adaptive Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 완료된 Dev18 ledger만 재사용해 비율별 Tier 대표를 주분석으로 봉인하고, PCA_LEGACY 참고선과 Tier 선 세 개만 담은 최종 그림을 만든다.

**Architecture:** 기존 `model_fixed`·`tier_fixed` 선택과 score ledger는 그대로 보존한다. 새 `tier_adaptive`는 holdout family 밖에서 한 번 고정한 fold recipe로 비율별 모델 점수를 계산하고, 최종 실행에는 full-panel `model_fixed` recipe를 연결한다. selection-only 경로는 score 배열을 읽거나 VUS evaluator를 실행하지 않는다. 봉인 evaluator·`ell_max` 신원만 대조한 뒤 ledger에서 정책 CSV·membership·그림을 다시 만든다.

**Tech Stack:** Python 3.11+, NumPy, Matplotlib, 표준 `csv/json/hashlib`, `unittest`

**Spec:** `docs/lead/dev18_ratio_adaptive_selection_design.md`

## Global Constraints

- `dev18_trial_score_ledger.csv` 1,602행을 재사용하며 모델 실행, HPO, VUS-PR 재계산과 원본 CSV 재처리를 하지 않는다.
- PCA_LEGACY는 adaptive 대표 후보나 성능·비용 gate가 아니다. q100 VUS-PR을 최종 그림의 수평 점선 참고선으로만 쓴다.
- 최종 `selection.png`에는 Tier별 선 세 개, PCA 점선, 작은 모델명, 축·범례만 둔다. 후보 점수·배제 근거·전환 내역은 CSV나 문서로 분리한다.
- 최종 정책은 기존 `model_fixed` config와 score variant를 쓴다. fold recipe도 모델의 전체 지원 비율에서 한 번만 고르며 비율마다 바꾸지 않는다.
- Tier 2의 5%는 다른 비율을 복사하지 않고 `unavailable`로 남긴다.
- 기존 budget, registry, score manifest, evaluator와 실험 점수 파일은 수정하지 않는다.
- 로컬에서는 모델·checkpoint forward, 실데이터 smoke, 전체 테스트와 PNG 렌더링을 실행하지 않는다.
- 기존 미추적 `tests/checks/run_colab_dev18_finish.ipynb`는 수정·추적하지 않는다.

---

### Task 1: Ledger-only ratio-adaptive selection

**Files:**
- Modify: `tests/ghl_main/run_dev18_tuning.py`
- Test: `tests/unit/test_dev18_tuning.py`

**Interfaces:**
- Produces: `load_trial_score_ledger(path, budget, *, expected_evaluator_sha256, expected_ell_max_id) -> list[dict]`
- Produces: `select_ratio_adaptive_policies(rows, registry, budget, *, model_fixed, evaluator_sha256) -> dict`
- Changes: `select_tuning_policies(...)` returns `tier_adaptive`, `adaptive_lofo`, `candidate_audit`, and `policy_transitions` in addition to existing keys.
- Preserves: existing `model_fixed`, `tier_fixed`, and fixed-policy CSV semantics.

- [ ] **Step 1: Write the failing ledger-loader test**

  Add a complete small CSV fixture with the real ledger header. Ratios and seeds are serialized strings. Assert that the loader returns integer `ratio`, integer `seed`, float `vus_pr`, rejects a duplicate logical key, rejects `status != complete`, rejects more than one evaluator SHA, and rejects a key set that differs from the supplied budget.

  ```python
  loaded = load_trial_score_ledger(
      path, budget,
      expected_evaluator_sha256="b" * 64,
      expected_ell_max_id="ell-v1",
  )
  self.assertIsInstance(loaded[0]["ratio"], int)
  self.assertIsInstance(loaded[0]["seed"], int)
  self.assertIsInstance(loaded[0]["vus_pr"], float)
  ```

- [ ] **Step 2: Run the loader test and verify RED**

  Run: `python -m unittest tests.unit.test_dev18_tuning.TestDev18Tuning.test_trial_score_ledger_loader_validates_sealed_logical_keys`

  Expected: import failure for `load_trial_score_ledger`.

- [ ] **Step 3: Implement the minimum typed loader**

  Validate the exact ledger header, complete status, unique `(series, model, config_id, ratio, seed, score_variant)`, one evaluator SHA, one `ell_max_id`, 18 series for the production budget, and the exact logical keys derived from `model_panels[].selected_config_ids`, `logical_ratios`, `seeds`, and `primary_score_variants`. Do not read score arrays, manifest rows, raw CSVs, or call `vus_pr`.

- [ ] **Step 4: Write the failing adaptive-selection test**

  Use a literal two-family fixture whose budget includes PCA_LEGACY, one unavailable model, and supported models with different q floors. Assert these behaviors independently:

  ```python
  adaptive = {
      (row["tier"], row["ratio"]): row
      for row in selection["tier_adaptive"]
  }
  self.assertEqual(adaptive[("t1", 100)]["selected_model"], "MWVAR")
  self.assertEqual(adaptive[("t2", 5)]["selection_status"], "unavailable")
  self.assertEqual(adaptive[("t2", 10)]["selected_model"], "GDN")
  self.assertEqual(adaptive[("t2", 40)]["selected_model"], "PaAno")
  self.assertNotIn("PCA_LEGACY", {
      row["selected_model"] for row in selection["tier_adaptive"]
  })
  ```

  Give one model two configs whose single-ratio winner differs from its all-supported-ratio winner. Assert every adaptive final row uses the full-panel `model_fixed` config, proving ratio-specific recipe reselection did not enter the final policy.

- [ ] **Step 5: Run the adaptive test and verify RED**

  Run: `python -m unittest tests.unit.test_dev18_tuning.TestDev18Tuning.test_ratio_adaptive_selection_uses_native_support_and_fixed_recipe`

  Expected: `tier_adaptive` is missing.

- [ ] **Step 6: Implement ratio-adaptive family-LOFO**

  For each non-PCA model with executed configs, choose each holdout fold recipe once over that model's full supported ratio set using the other families. Evaluate those fold recipes at each supported ratio. Select the highest model mean per `(tier, ratio)` with the existing tolerance and tuple tie-break. Use the full-panel `model_fixed` config in the final row. Emit one unavailable row when a Tier has no candidate.

- [ ] **Step 7: Emit candidate audit and transitions**

  Candidate rows must contain `tier, ratio, model, config_id, score_variant, eligibility, family_lofo_vus_pr, selected, reason_code, reason`. PCA rows use `reference_only`; no-config rows use `full_panel_config_unavailable`; unsupported ratios use `ratio_unsupported`. Transition rows must contain previous/current ratio, model and config, `initial|keep|switch|unavailable`, plus `transition_key="tr" + sha256(canonical_json)[:12]`.

- [ ] **Step 8: Run Task 1 tests and verify GREEN**

  Run: `python -m unittest tests.unit.test_dev18_tuning.TestDev18Tuning.test_trial_score_ledger_loader_validates_sealed_logical_keys tests.unit.test_dev18_tuning.TestDev18Tuning.test_ratio_adaptive_selection_uses_native_support_and_fixed_recipe`

  Expected: both tests pass without reading the user's Downloads files.

- [ ] **Step 9: Commit Task 1**

  ```bash
  git add tests/ghl_main/run_dev18_tuning.py tests/unit/test_dev18_tuning.py
  git commit -m "feat: select Dev18 tier models by ratio"
  ```

### Task 2: Adaptive membership and site-input contracts

**Files:**
- Modify: `tests/ghl_main/run_dev18_tuning.py`
- Modify: `src/common/load_final_membership.py`
- Modify: `tests/unit/test_final_membership.py`
- Create: `configs/deployment_scenario.schema.json`
- Test: `tests/unit/test_dev18_tuning.py`

**Interfaces:**
- Changes: membership `analysis_kind` accepts and requires `tier_adaptive`.
- Preserves: membership columns and physical execution key `(model, config_id, physical_ratio)`.
- Produces: `tier_ratio_candidate_audit.csv`, `tier_policy_transitions.csv`, `ratio_adaptive_selection.csv`.
- Produces: a JSON Schema with nullable site-entered values and no numeric defaults.

- [ ] **Step 1: Write the failing membership completeness test**

  Extend the membership fixture with one `tier_adaptive` row per Tier·ratio·split. Assert that omitting any adaptive Tier·ratio fails and that all three kinds are required.

  ```python
  self.assertEqual(
      {row["analysis_kind"] for row in membership},
      {"model_fixed", "tier_fixed", "tier_adaptive"},
  )
  ```

- [ ] **Step 2: Write the failing physical-union invariance test**

  Build the union once with adaptive rows and once without them. Use only adaptive model/config/ratio keys already present in `model_fixed` and require literal equality. This catches accidental new model execution caused by the logical policy.

- [ ] **Step 3: Run membership tests and verify RED**

  Run: `python -m unittest tests.unit.test_final_membership`

  Expected: `tier_adaptive` is rejected or not required.

- [ ] **Step 4: Implement the minimum membership extension**

  Add `tier_adaptive` to `ANALYSIS_KINDS`, require a complete Tier·ratio grid for it, and extend `build_final_membership_rows()`. For an unavailable adaptive ratio, use the Tier's fixed representative only as the schema-valid non-running model/config placeholder; keep `physical_ratio` empty and preserve the adaptive unavailability reason.

- [ ] **Step 5: Write policy CSVs without crowding the figure**

  Extend `write_selection_reports(..., write_plots=True)` so `write_plots=False` writes all CSVs without creating a Matplotlib figure. Write `ratio_adaptive_selection.csv`, `tier_ratio_candidate_audit.csv`, and `tier_policy_transitions.csv` from Task 1's rows. Keep detailed candidate and exclusion information out of `selection.png`.

- [ ] **Step 6: Add the field-input JSON Schema**

  The root requires `transition_key` and exposes nullable fields for `currency`, analysis horizon, observation, training, inference, memory, artifact, false-alarm, miss, transition validation, deployment and downtime costs, plus optional resource/performance constraints. Do not use `default`. `0` remains a valid explicit number; `null` means missing and must not be interpreted as zero by a future consumer.

- [ ] **Step 7: Add the schema contract test**

  Load the schema with `json.loads`, recursively assert that no node has a `default` key, and assert the required `transition_key` and nullable `model_switch_cost`, `validation_cost`, and `deployment_cost` properties exist. The expected field names are literals in the test.

- [ ] **Step 8: Run Task 2 tests and verify GREEN**

  Run: `python -m unittest tests.unit.test_final_membership tests.unit.test_dev18_tuning.TestDev18Tuning.test_reports_separate_adaptive_audit_and_transition_tables tests.unit.test_dev18_tuning.TestDev18Tuning.test_deployment_scenario_schema_has_no_cost_defaults`

- [ ] **Step 9: Commit Task 2**

  ```bash
  git add src/common/load_final_membership.py tests/ghl_main/run_dev18_tuning.py tests/unit/test_final_membership.py tests/unit/test_dev18_tuning.py configs/deployment_scenario.schema.json
  git commit -m "feat: expose adaptive policy and site cost inputs"
  ```

### Task 3: Clean final Tier plot and font resolution

**Files:**
- Modify: `tests/ghl_main/run_dev18_tuning.py`
- Test: `tests/unit/test_dev18_tuning.py`

**Interfaces:**
- Produces: `build_ratio_adaptive_plot_data(rows, selection) -> dict`
- Produces: `configure_plot_font() -> dict`
- Changes: `selection.png` becomes one clean axis with three Tier curves and one PCA reference line.

- [ ] **Step 1: Write the failing pure plot-data test**

  Build a literal selection with three Tier curves and one unavailable point. Assert exact sorted tuples and the PCA reference value derived from PCA_LEGACY's selected q100 recipe.

  ```python
  data = build_ratio_adaptive_plot_data(rows, selection)
  self.assertEqual(
      [(row["tier"], row["ratio"], row["model"]) for row in data["points"]],
      [("t1", 5, "MWVAR"), ("t1", 100, "MWVAR"),
       ("t2", 40, "PaAno"), ("t3", 5, "TSPulse")],
  )
  self.assertEqual(data["unavailable"], [{"tier": "t2", "ratio": 5}])
  self.assertAlmostEqual(data["pca_reference_vus_pr"], 0.283669)
  ```

- [ ] **Step 2: Run the plot-data test and verify RED**

  Run: `python -m unittest tests.unit.test_dev18_tuning.TestDev18Tuning.test_final_plot_data_contains_only_tier_paths_and_pca_reference`

  Expected: import failure for `build_ratio_adaptive_plot_data`.

- [ ] **Step 3: Implement the pure plot-data builder**

  Use adaptive `selection_score` for the Tier y-values. Read the PCA reference from the ledger's q100 family-macro score for its full-panel selected config. Return only plot points, unavailable positions and the reference value; keep candidate details in Task 2 CSVs.

- [ ] **Step 4: Write the failing font-resolution test**

  Patch environment and file discovery without rendering. Require an explicit `TSAD_KOREAN_FONT_PATH` to win, Windows Malgun Gothic to be next, installed NanumGothic to be the fallback, and the default Matplotlib family to remain valid when none exists. Invalid explicit paths must raise a clear error.

- [ ] **Step 5: Implement deterministic font setup**

  Register the selected font before setting `matplotlib.rcParams["font.family"]`. Use Malgun Gothic when available and NanumGothic on Lightning when installed. Set `axes.unicode_minus=False`. The final plot uses ASCII labels so the absence of a Korean font never creates broken plot text.

- [ ] **Step 6: Replace only the final selection figure**

  Plot one line per Tier, light horizontal grid lines, exact ratio ticks, and one gray `--` PCA reference. Annotate every available marker with only its selected model at `fontsize=7`; place Tier 1 and Tier 3 labels on opposite vertical offsets to prevent overlap. Mark Tier 2 q5 with a small gray `unavailable`. Do not add a table, candidate curves, long reasons, cost components, shaded panels, or extra axes.

- [ ] **Step 7: Run Task 3 tests and verify GREEN**

  Run: `python -m unittest tests.unit.test_dev18_tuning.TestDev18Tuning.test_final_plot_data_contains_only_tier_paths_and_pca_reference tests.unit.test_dev18_tuning.TestDev18Tuning.test_plot_font_resolution_prefers_malgun_and_accepts_lightning_fallback`

  Expected: both tests pass without saving a PNG.

- [ ] **Step 8: Commit Task 3**

  ```bash
  git add tests/ghl_main/run_dev18_tuning.py tests/unit/test_dev18_tuning.py
  git commit -m "feat: draw clean adaptive tier selection plot"
  ```

### Task 4: Selection-only Lightning handoff and project decisions

**Files:**
- Modify: `tests/checks/finish_lightning_dev18.py`
- Modify: `tests/ghl_main/run_dev18_tuning.py`
- Modify: `tests/unit/test_dev18_tuning.py`
- Modify: `docs/lead/plan_v5.md`
- Modify: `docs/lead/decisions.md`
- Modify: `docs/lead/next_session.md`
- Modify: `docs/lead/lightning_studio.md`
- Modify: `docs/role_B/score_interface.md`

**Interfaces:**
- Produces: `finish_selection_from_ledger(ledger_path, *, output_directory=DEFAULT_RESULT_DIRECTORY) -> dict`
- Changes: `finish_lightning_dev18.py --selection-only --ledger PATH --result-directory PATH` skips scoring and writes a separate selection receipt.

- [ ] **Step 1: Write the failing selection-only test**

  Patch `build_trial_score_ledger` and `vus_pr` to raise if called. Supply a small sealed ledger fixture and patched registry/budget. Call `finish_selection_from_ledger()` and assert it writes the three adaptive CSVs and final membership while neither expensive function is invoked.

- [ ] **Step 2: Run the selection-only test and verify RED**

  Run: `python -m unittest tests.unit.test_dev18_tuning.TestDev18Tuning.test_selection_only_reuses_ledger_without_vus_or_score_arrays`

  Expected: import failure for `finish_selection_from_ledger`.

- [ ] **Step 3: Implement the selection-only path**

  Load and validate the current registry and budget, call Task 1's loader and selectors, write policies, membership, CSV reports and plots, and return ledger SHA, row count, membership SHA and selected path. In the CLI, `--selection-only` must bypass score-manifest hashing, checkpoint locking, completion deletion and `finish_tuning()`. Write `selection_complete.json`, not `finish_complete.json`.

- [ ] **Step 4: Run the selection-only and direct CLI tests**

  Run: `python -m unittest tests.unit.test_dev18_tuning.TestDev18Tuning.test_selection_only_reuses_ledger_without_vus_or_score_arrays tests.unit.test_dev18_tuning.TestDev18Tuning.test_direct_file_cli_can_import_project_packages`

  Expected: both pass without score files.

- [ ] **Step 5: Update the research decision and handoff documents**

  Replace the old statement that `tier_fixed` is the sole Tier primary analysis. Record `tier_adaptive` as the operational primary, retain `tier_fixed` as the controlled comparator, keep PCA as plot-only reference, and state that adaptive membership adds no physical execution beyond `model_fixed`. Correct final membership count to 294 rows. Add the exact Lightning selection-only command and expected output files. Preserve the old resource gate as historical execution evidence rather than rerunning it.

- [ ] **Step 6: Run document consistency checks**

  Run: `rg -n "tier_adaptive|selection-only|294|PCA_LEGACY|reference" docs/lead docs/role_B`

  Expected: primary/comparator roles, row count, command and PCA role agree.

- [ ] **Step 7: Commit Task 4**

  ```bash
  git add tests/checks/finish_lightning_dev18.py tests/ghl_main/run_dev18_tuning.py tests/unit/test_dev18_tuning.py docs/lead docs/role_B/score_interface.md
  git commit -m "feat: rebuild Dev18 selection from completed ledger"
  ```

### Task 5: Focused verification and handoff

**Files:**
- Review: all files changed by Tasks 1–4

**Interfaces:**
- Produces: verified local code and one exact Lightning command for remote PNG generation.

- [ ] **Step 1: Run the affected lightweight tests**

  Run: `python -m unittest tests.unit.test_dev18_tuning tests.unit.test_final_membership`

  Expected: all tests pass without model forward, VUS computation, raw CSV reads or PNG saves.

- [ ] **Step 2: Run static checks**

  Run: `python -m compileall tests/ghl_main/run_dev18_tuning.py tests/checks/finish_lightning_dev18.py src/common/load_final_membership.py`

  Run: `git diff --check`

- [ ] **Step 3: Verify sealed execution inputs did not change**

  Run: `git diff 83232f1 -- configs/model_registry.yaml experiments/01_ghl_main/snapshots/dev18_selection/dev18_budget_manifest.json experiments/01_ghl_main/logs/dev18_score_manifest.csv`

  Expected: no diff.

- [ ] **Step 4: Verify scope and user files**

  Run: `git status --short --untracked-files=all`

  Confirm the existing untracked notebook remains unmodified and no experiment output, cache, font binary or downloaded data is tracked.

- [ ] **Step 5: Review the complete diff**

  Compare every requirement in `docs/lead/dev18_ratio_adaptive_selection_design.md` with the implementation. Confirm `selection.png` has no data table or auxiliary subplot and every detailed item has a CSV or Markdown home.
