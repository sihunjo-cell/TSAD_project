-- ERDCloud SQL Import / MySQL 8.0.16+ / 2026-09-09
-- 엑셀 중심 간소화안: 핵심 6개 표 + 모델별 파라미터 8개 표.
-- 실험 하나당 DB 하나. 현재 SQLite를 변경하는 migration이 아니다.

CREATE TABLE experiment_info (
    id INT NOT NULL DEFAULT 1,
    experiment_name VARCHAR(128) NOT NULL COMMENT '이번 실험 이름',
    identity_json JSON NOT NULL COMMENT '기존 실험 신원: 예산, 코드, 입력, 저장 버전',
    environment_json JSON NULL COMMENT '확인한 실행 환경',
    evidence_files_json JSON NOT NULL COMMENT '제외 원표, 선택, LOFO, 최종 요청 파일의 경로와 SHA',
    PRIMARY KEY (id),
    CONSTRAINT ck_experiment_single CHECK (id = 1)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='실험 정보 한 행';

CREATE TABLE prefix_features (
    prefix_feature_id VARCHAR(65) NOT NULL COMMENT '엑셀: 입력 특징 식별ID',
    csv_id VARCHAR(512) NOT NULL COMMENT '출처 경로를 포함한 파일 신원',
    csv_file VARCHAR(512) NOT NULL COMMENT '엑셀: CSV_file',
    q_percent INT NOT NULL COMMENT '엑셀: q비율',
    training_boundary BIGINT NOT NULL COMMENT '엑셀: 정상 학습 구간 N',
    observed_row BIGINT NOT NULL COMMENT '엑셀: 현재 q에서 관측한 행 수',
    input_column INT NOT NULL COMMENT '엑셀: 입력 센서 수',
    constant_channel_count INT NULL COMMENT '엑셀: 상수채널수',
    channel_std_median DOUBLE NULL COMMENT '엑셀: 채널 std 중앙값',
    channel_acf_lag1_median DOUBLE NULL COMMENT '엑셀: 채널 ACF lag1 중앙값',
    absolute_correlation_median DOUBLE NULL COMMENT '엑셀: 채널간 절대 상관 중앙값',
    channel_interquartile_range_median DOUBLE NULL COMMENT '채널 IQR 중앙값',
    channel_difference_q90_iqr_ratio_median DOUBLE NULL COMMENT '채널 차분 절댓값 Q90/IQR 중앙값',
    channel_median_shift_iqr_ratio_median DOUBLE NULL COMMENT '채널 전후반 중앙값 이동/IQR 중앙값',
    channel_spectral_entropy_median DOUBLE NULL COMMENT '채널 정규화 스펙트럼 엔트로피 중앙값',
    feature_details_json JSON NOT NULL COMMENT '원본 신원, 채널 순서, 범위, 산식, 유효 개수, NULL 사유',
    PRIMARY KEY (prefix_feature_id),
    UNIQUE KEY uq_prefix_file_q (csv_id, q_percent),
    UNIQUE KEY uq_prefix_file (prefix_feature_id, csv_id),
    CONSTRAINT ck_prefix_q CHECK (q_percent IN (5,10,20,40,60,80,100)),
    CONSTRAINT ck_prefix_shape CHECK (training_boundary > 0 AND observed_row >= 0 AND observed_row <= FLOOR(training_boundary * (q_percent / 100.0)) AND input_column > 0),
    CONSTRAINT ck_prefix_statistics CHECK (constant_channel_count >= 0 AND constant_channel_count <= input_column AND channel_std_median >= 0 AND absolute_correlation_median >= 0),
    CONSTRAINT ck_prefix_extended_statistics CHECK (channel_interquartile_range_median >= 0 AND channel_difference_q90_iqr_ratio_median >= 0 AND channel_median_shift_iqr_ratio_median >= 0 AND channel_spectral_entropy_median BETWEEN 0 AND 1)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='입력 특징: CSV 하나의 q-prefix';

CREATE TABLE channel_features (
    prefix_feature_id VARCHAR(65) NOT NULL,
    channel_index INT NOT NULL COMMENT '0부터 시작하는 센서 순서',
    channel_name VARCHAR(255) NOT NULL COMMENT '엑셀: channel(col)',
    mean DOUBLE NULL COMMENT '평균',
    std DOUBLE NULL COMMENT '표준편차',
    median DOUBLE NULL COMMENT '중앙값',
    acf_lag1 DOUBLE NULL COMMENT 'lag1 자기상관',
    interquartile_range DOUBLE NULL COMMENT '유한값의 Q75-Q25',
    difference_q90_iqr_ratio DOUBLE NULL COMMENT '인접 차분 절댓값 Q90/IQR',
    median_shift_iqr_ratio DOUBLE NULL COMMENT '전후반 중앙값 차이 절댓값/IQR',
    spectral_entropy DOUBLE NULL COMMENT 'DC 제외 양의 주파수 rfft 전력의 정규화 Shannon 엔트로피',
    channel_details_json JSON NOT NULL COMMENT '유효값 수, 상수 여부, 통계별 NULL 사유',
    PRIMARY KEY (prefix_feature_id, channel_index),
    CONSTRAINT fk_channel_prefix FOREIGN KEY (prefix_feature_id) REFERENCES prefix_features (prefix_feature_id),
    CONSTRAINT ck_channel_values CHECK (channel_index >= 0 AND std >= 0),
    CONSTRAINT ck_channel_extended_statistics CHECK (interquartile_range >= 0 AND difference_q90_iqr_ratio >= 0 AND median_shift_iqr_ratio >= 0 AND spectral_entropy BETWEEN 0 AND 1)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='채널 특징: prefix 안의 센서 하나';

CREATE TABLE model_configs (
    config_id VARCHAR(13) NOT NULL COMMENT '엑셀: 전체설정 식별ID. 기존 ID 유지',
    model VARCHAR(64) NOT NULL COMMENT '엑셀: 실행모델',
    candidate_order INT NOT NULL COMMENT '엑셀: 후보번호. 검토 순서이며 PK가 아님',
    tier VARCHAR(2) NOT NULL,
    target_use VARCHAR(32) NOT NULL,
    deterministic BOOLEAN NOT NULL,
    settings_json JSON NOT NULL COMMENT '원본 전체 설정: 파라미터, 소스, checkpoint, 전후처리',
    PRIMARY KEY (config_id),
    UNIQUE KEY uq_config_model (config_id, model),
    CONSTRAINT ck_config_values CHECK (candidate_order > 0 AND tier IN ('t1','t2','t3') AND deterministic IN (0,1)),
    CONSTRAINT ck_config_target CHECK (target_use IN ('training_free','fit_full_prefix','strict_zero_shot'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='모델 설정: 확정된 후보 한 벌';

CREATE TABLE model_runs (
    physical_execution_id VARCHAR(32) NOT NULL COMMENT '엑셀: 실제 실행 ID. 기존 실행 이력의 run_id',
    csv_id VARCHAR(512) NOT NULL,
    config_id VARCHAR(13) NOT NULL,
    seed INT NOT NULL,
    physical_q INT NOT NULL COMMENT '실제 실행 q. 입력과 무관한 모델은 재사용 표시 100',
    attempt_index INT NOT NULL COMMENT '같은 조합의 재시도 번호',
    status VARCHAR(32) NOT NULL COMMENT '실제 실행 시도의 상태. 채점 상태와 구분',
    started_at DATETIME(6) NOT NULL COMMENT 'UTC',
    finished_at DATETIME(6) NULL COMMENT 'UTC. 확인하지 못한 종료시간은 NULL',
    elapsed_seconds DOUBLE NULL COMMENT '실행 시도 전체 시간',
    training_patch_count BIGINT NULL,
    memory_count BIGINT NULL COMMENT 'PaAno 실제 memory 수',
    epochs_completed INT NULL,
    selected_epoch INT NULL,
    optimizer_updates BIGINT NULL,
    training_seconds DOUBLE NULL,
    inference_seconds DOUBLE NULL,
    peak_gpu_memory_bytes BIGINT NULL,
    run_details_json JSON NOT NULL COMMENT '원본 이력, snapshot, checkpoint와 scaler 참조, 측정 범위, NULL 사유',
    PRIMARY KEY (physical_execution_id),
    UNIQUE KEY uq_run_attempt (csv_id, config_id, physical_q, seed, attempt_index),
    UNIQUE KEY uq_run_context (physical_execution_id, csv_id, config_id, seed),
    CONSTRAINT fk_run_config FOREIGN KEY (config_id) REFERENCES model_configs (config_id),
    CONSTRAINT ck_run_identity CHECK (seed >= 0 AND attempt_index >= 0 AND physical_q IN (5,10,20,40,60,80,100)),
    CONSTRAINT ck_run_status CHECK (status IN ('running','complete','failed','interrupted','timeout')),
    CONSTRAINT ck_run_time CHECK (finished_at IS NULL OR finished_at >= started_at),
    CONSTRAINT ck_run_measurements CHECK (elapsed_seconds >= 0 AND training_patch_count >= 0 AND memory_count >= 0 AND epochs_completed >= 0 AND selected_epoch >= 0 AND optimizer_updates >= 0 AND training_seconds >= 0 AND inference_seconds >= 0 AND peak_gpu_memory_bytes >= 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='실행 기록: 실제 시도 한 번과 비용';

CREATE TABLE results (
    prefix_feature_id VARCHAR(65) NOT NULL,
    config_id VARCHAR(13) NOT NULL,
    seed INT NOT NULL,
    score_variant VARCHAR(16) NOT NULL COMMENT '일반 모델은 빈 문자열. TSPulse는 time, fft, pred, ensemble',
    csv_id VARCHAR(512) NOT NULL COMMENT '입력과 실행이 같은 파일인지 검사하는 연결 열',
    physical_execution_id VARCHAR(32) NULL COMMENT '실제로 확인한 run_id만 참조',
    primary_score BOOLEAN NOT NULL,
    status VARCHAR(32) NOT NULL COMMENT '엑셀: 완료여부. 실행 후 채점 전 상태도 구분',
    status_reason TEXT NULL,
    vus_pr DOUBLE NULL COMMENT '엑셀: 평가점수. 미채점 또는 실패는 NULL',
    score_file TEXT NULL COMMENT '엑셀: 점수 파일',
    score_sha256 CHAR(64) NULL COMMENT '엑셀: 점수 SHA-256',
    metadata_file TEXT NULL COMMENT 'raw, smoothed, 채널 배열과 정렬 범위의 원본 참조',
    metadata_sha256 CHAR(64) NULL,
    evaluator_sha256 CHAR(64) NULL,
    ell_max_id VARCHAR(128) NULL,
    manifest_reference TEXT NULL,
    ledger_reference TEXT NULL,
    source_reference_json JSON NULL COMMENT '기존 실패 합성 ID와 training_group_id 등 출처값',
    PRIMARY KEY (prefix_feature_id, config_id, seed, score_variant),
    CONSTRAINT fk_result_prefix FOREIGN KEY (prefix_feature_id, csv_id) REFERENCES prefix_features (prefix_feature_id, csv_id),
    CONSTRAINT fk_result_config FOREIGN KEY (config_id) REFERENCES model_configs (config_id),
    CONSTRAINT fk_result_run FOREIGN KEY (physical_execution_id, csv_id, config_id, seed) REFERENCES model_runs (physical_execution_id, csv_id, config_id, seed),
    CONSTRAINT ck_result_identity CHECK (seed >= 0 AND primary_score IN (0,1) AND score_variant IN ('','time','fft','pred','ensemble')),
    CONSTRAINT ck_result_status CHECK (status IN ('complete','executed_pending_score','executed_unscored','failed','interrupted','running','timeout')),
    CONSTRAINT ck_result_score CHECK (
        (status = 'complete' AND primary_score = 1 AND vus_pr IS NOT NULL AND vus_pr BETWEEN 0 AND 1
         AND evaluator_sha256 IS NOT NULL AND evaluator_sha256 REGEXP '^[0-9a-f]{64}$'
         AND ell_max_id IS NOT NULL AND CHAR_LENGTH(TRIM(ell_max_id)) > 0
         AND ledger_reference IS NOT NULL AND CHAR_LENGTH(TRIM(ledger_reference)) > 0)
        OR (status <> 'complete' AND vus_pr IS NULL)),
    CONSTRAINT ck_result_files CHECK (
        status NOT IN ('complete','executed_pending_score','executed_unscored')
        OR (physical_execution_id IS NOT NULL AND CHAR_LENGTH(TRIM(physical_execution_id)) > 0
            AND score_file IS NOT NULL AND CHAR_LENGTH(TRIM(score_file)) > 0
            AND score_sha256 IS NOT NULL AND score_sha256 REGEXP '^[0-9a-f]{64}$'
            AND metadata_file IS NOT NULL AND CHAR_LENGTH(TRIM(metadata_file)) > 0
            AND metadata_sha256 IS NOT NULL AND metadata_sha256 REGEXP '^[0-9a-f]{64}$'
            AND manifest_reference IS NOT NULL AND CHAR_LENGTH(TRIM(manifest_reference)) > 0))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='성능 결과: CSV, q, 설정, seed, head별 한 행';

-- 모델별 상세 표: 엑셀의 파라미터를 숫자와 Boolean 열로 유지한다.
-- model은 부모 설정과 모델 종류가 같은지 검사하는 복합 FK의 일부다.

CREATE TABLE config_mwvar (
    config_id VARCHAR(13) NOT NULL,
    model VARCHAR(64) NOT NULL,
    `window` INT NOT NULL,
    centered BOOLEAN NOT NULL,
    ddof INT NOT NULL,
    PRIMARY KEY (config_id),
    CONSTRAINT fk_mwvar_config FOREIGN KEY (config_id, model) REFERENCES model_configs (config_id, model),
    CONSTRAINT ck_mwvar_model CHECK (model = 'MWVAR'),
    CONSTRAINT ck_mwvar_values CHECK (`window` >= 2 AND centered IN (0,1) AND ddof >= 0 AND ddof < `window`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='MWVAR 파라미터';

CREATE TABLE config_sqdiff (
    config_id VARCHAR(13) NOT NULL,
    model VARCHAR(64) NOT NULL,
    `lag` INT NULL,
    `window` INT NOT NULL,
    centered BOOLEAN NULL,
    correction DOUBLE NOT NULL,
    PRIMARY KEY (config_id),
    CONSTRAINT fk_sqdiff_config FOREIGN KEY (config_id, model) REFERENCES model_configs (config_id, model),
    CONSTRAINT ck_sqdiff_model CHECK (
        (model IN ('SQDIFF_LAST1','SQDIFF_LAST3') AND `lag` IS NOT NULL AND `lag` > 0 AND `window` = `lag` + 1 AND centered IS NULL)
        OR (model = 'SQDIFF_CENTERED5' AND `lag` IS NULL AND `window` = 5 AND centered IS NOT NULL AND centered = 1)),
    CONSTRAINT ck_sqdiff_correction CHECK (correction > 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='세 SQDIFF 모델 파라미터';

CREATE TABLE config_one_liner_ensemble (
    config_id VARCHAR(13) NOT NULL,
    model VARCHAR(64) NOT NULL,
    variance_window INT NOT NULL,
    difference_window INT NOT NULL,
    difference_centered BOOLEAN NOT NULL,
    difference_correction DOUBLE NOT NULL,
    PRIMARY KEY (config_id),
    CONSTRAINT fk_ensemble_config FOREIGN KEY (config_id, model) REFERENCES model_configs (config_id, model),
    CONSTRAINT ck_ensemble_model CHECK (
        (model = 'MWVAR96_SQDIFF_LAST3' AND difference_window = 4 AND difference_centered = 0)
        OR (model = 'MWVAR96_SQDIFF_CENTERED5' AND difference_window = 5 AND difference_centered = 1)),
    CONSTRAINT ck_ensemble_values CHECK (variance_window = 96 AND difference_correction > 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='두 One-Liner 앙상블 파라미터';

CREATE TABLE config_pca (
    config_id VARCHAR(13) NOT NULL,
    model VARCHAR(64) NOT NULL,
    n_components_mode VARCHAR(24) NOT NULL,
    n_components_fraction DOUBLE NULL COMMENT 'all이면 NULL. 미수집과 구분',
    `window` INT NOT NULL,
    zero_pruning BOOLEAN NOT NULL,
    PRIMARY KEY (config_id),
    CONSTRAINT fk_pca_config FOREIGN KEY (config_id, model) REFERENCES model_configs (config_id, model),
    CONSTRAINT ck_pca_model CHECK (model = 'PCA_LEGACY'),
    CONSTRAINT ck_pca_components CHECK (
        (n_components_mode = 'all' AND n_components_fraction IS NULL)
        OR (n_components_mode = 'variance_fraction' AND n_components_fraction IS NOT NULL AND n_components_fraction > 0 AND n_components_fraction < 1)),
    CONSTRAINT ck_pca_values CHECK (`window` > 0 AND zero_pruning IN (0,1))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='PCA 파라미터';

CREATE TABLE config_paano (
    config_id VARCHAR(13) NOT NULL,
    model VARCHAR(64) NOT NULL,
    patch_size INT NOT NULL,
    learning_rate DOUBLE NOT NULL,
    iterations INT NOT NULL,
    batch_size INT NOT NULL,
    weight_decay DOUBLE NOT NULL,
    memory_fraction DOUBLE NOT NULL,
    memory_seed INT NOT NULL,
    neighbors INT NOT NULL,
    use_revin BOOLEAN NOT NULL,
    memory_policy VARCHAR(32) NOT NULL,
    optimizer VARCHAR(32) NOT NULL,
    embedding_dimension INT NOT NULL,
    checkpoint_selection VARCHAR(64) NOT NULL,
    PRIMARY KEY (config_id),
    CONSTRAINT fk_paano_config FOREIGN KEY (config_id, model) REFERENCES model_configs (config_id, model),
    CONSTRAINT ck_paano_model CHECK (model = 'PaAno'),
    CONSTRAINT ck_paano_values CHECK (patch_size > 0 AND learning_rate > 0 AND iterations > 0 AND batch_size >= 2 AND weight_decay >= 0 AND memory_fraction > 0 AND memory_fraction < 1 AND memory_seed >= 0 AND neighbors > 0 AND use_revin IN (0,1) AND embedding_dimension > 0),
    CONSTRAINT ck_paano_recipe CHECK (memory_policy = 'official_minimum' AND optimizer = 'AdamW' AND checkpoint_selection = 'best_training_loss')
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='PaAno 파라미터와 고정 처리';

CREATE TABLE config_gdn (
    config_id VARCHAR(13) NOT NULL,
    model VARCHAR(64) NOT NULL,
    embedding INT NOT NULL,
    hidden INT NOT NULL,
    hidden_is_active BOOLEAN NOT NULL,
    topk INT NOT NULL,
    batch_size INT NOT NULL,
    epochs INT NOT NULL,
    patience INT NOT NULL,
    validation_ratio DOUBLE NOT NULL,
    optimizer_beta1 DOUBLE NOT NULL,
    optimizer_beta2 DOUBLE NOT NULL,
    `window` INT NOT NULL,
    stride INT NOT NULL,
    learning_rate DOUBLE NOT NULL,
    weight_decay DOUBLE NOT NULL,
    out_layer_num INT NOT NULL,
    graph_heads INT NOT NULL,
    graph_dropout DOUBLE NOT NULL,
    output_dropout DOUBLE NOT NULL,
    optimizer VARCHAR(32) NOT NULL,
    PRIMARY KEY (config_id),
    CONSTRAINT fk_gdn_config FOREIGN KEY (config_id, model) REFERENCES model_configs (config_id, model),
    CONSTRAINT ck_gdn_model CHECK (model = 'GDN'),
    CONSTRAINT ck_gdn_hidden CHECK ((out_layer_num = 1 AND hidden_is_active = 0) OR (out_layer_num > 1 AND hidden_is_active = 1)),
    CONSTRAINT ck_gdn_values CHECK (embedding > 0 AND hidden > 0 AND topk > 0 AND batch_size > 0 AND epochs > 0 AND patience > 0 AND `window` > 0 AND stride > 0 AND learning_rate > 0 AND weight_decay >= 0 AND graph_heads > 0 AND validation_ratio > 0 AND validation_ratio < 1 AND optimizer_beta1 >= 0 AND optimizer_beta1 < 1 AND optimizer_beta2 >= 0 AND optimizer_beta2 < 1 AND graph_dropout >= 0 AND graph_dropout < 1 AND output_dropout >= 0 AND output_dropout < 1)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='GDN의 정해진 세 조합';

CREATE TABLE config_timercd (
    config_id VARCHAR(13) NOT NULL,
    model VARCHAR(64) NOT NULL,
    context_length INT NOT NULL,
    checkpoint_variant VARCHAR(32) NOT NULL,
    score_head VARCHAR(32) NOT NULL,
    PRIMARY KEY (config_id),
    CONSTRAINT fk_timercd_config FOREIGN KEY (config_id, model) REFERENCES model_configs (config_id, model),
    CONSTRAINT ck_timercd_model CHECK (model = 'TimeRCD'),
    CONSTRAINT ck_timercd_values CHECK (context_length > 0 AND checkpoint_variant = 'multi' AND score_head = 'probability')
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='TimeRCD 파라미터';

CREATE TABLE config_tspulse (
    config_id VARCHAR(13) NOT NULL,
    model VARCHAR(64) NOT NULL,
    aggregation_window INT NOT NULL,
    context_length INT NOT NULL,
    patch_size INT NOT NULL,
    supports_time BOOLEAN NOT NULL,
    supports_fft BOOLEAN NOT NULL,
    supports_pred BOOLEAN NOT NULL,
    supports_ensemble BOOLEAN NOT NULL,
    inference_batch_size INT NOT NULL,
    native_smoothing_window INT NOT NULL,
    PRIMARY KEY (config_id),
    CONSTRAINT fk_tspulse_config FOREIGN KEY (config_id, model) REFERENCES model_configs (config_id, model),
    CONSTRAINT ck_tspulse_model CHECK (model = 'TSPulse'),
    CONSTRAINT ck_tspulse_heads CHECK (supports_time = 1 AND supports_fft = 1 AND supports_pred = 1 AND supports_ensemble = 1),
    CONSTRAINT ck_tspulse_values CHECK (aggregation_window > 0 AND context_length > 0 AND patch_size > 0 AND inference_batch_size > 0 AND native_smoothing_window > 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='TSPulse 파라미터';
