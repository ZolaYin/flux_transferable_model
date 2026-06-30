import os
import torch


def get_carbonbench_flux_from_scratch_config():
    config = {}

    data_root = os.environ.get('CARBONBENCH_DATA_ROOT', '../flux_data/carbonbench')
    split_file = os.environ.get('CARBONBENCH_SPLIT_FILE', os.path.join(data_root, 'split_enf_C_to_D_min1095.csv'))
    patch_manifest_file = os.environ.get(
        'CARBONBENCH_PATCH_MANIFEST_FILE',
        os.path.join(data_root, 'patch_manifest_enf_C_to_D_hls_full_v1_downloaded.csv')
    )
    restrict_to_patch_manifest_rows = os.environ.get('CARBONBENCH_RESTRICT_TO_PATCH_MANIFEST', '0') == '1'
    restrict_to_patch_manifest_sites = os.environ.get('CARBONBENCH_RESTRICT_TO_PATCH_MANIFEST_SITES', '0') == '1'
    use_image_branch = os.environ.get('CARBONBENCH_USE_IMAGE_BRANCH', '0') == '1'
    include_patch_fields = os.environ.get(
        'CARBONBENCH_INCLUDE_PATCH_FIELDS',
        '1' if use_image_branch else '0'
    ) == '1'
    use_sequence_branch = os.environ.get('CARBONBENCH_USE_SEQUENCE_BRANCH', '0') == '1'
    igbp_filter = os.environ.get('CARBONBENCH_IGBP_FILTER', 'forest')
    feature_set_name = os.environ.get('CARBONBENCH_FEATURE_SET_NAME', 'standard')
    image_encoder_type = os.environ.get('CARBONBENCH_IMAGE_ENCODER_TYPE', 'resnet18')
    image_resnet_variant = os.environ.get('CARBONBENCH_IMAGE_RESNET_VARIANT', 'resnet18')
    image_resnet_pretrained = os.environ.get('CARBONBENCH_IMAGE_RESNET_PRETRAINED', '0') == '1'
    image_branch_trainable = os.environ.get('CARBONBENCH_IMAGE_BRANCH_TRAINABLE', '1') == '1'
    image_context_mode = os.environ.get('CARBONBENCH_IMAGE_CONTEXT_MODE', 'exact')
    image_context_max_patches = int(os.environ.get('CARBONBENCH_IMAGE_CONTEXT_MAX_PATCHES', '1'))
    fusion_mode = os.environ.get('CARBONBENCH_FUSION_MODE', 'concat')
    image_gate_init_bias = float(os.environ.get('CARBONBENCH_IMAGE_GATE_INIT_BIAS', '-2.0'))
    sequence_length = int(os.environ.get('CARBONBENCH_SEQUENCE_LENGTH', '32'))
    sequence_include_current = os.environ.get('CARBONBENCH_SEQUENCE_INCLUDE_CURRENT', '0') == '1'
    sequence_encoder_type = os.environ.get('CARBONBENCH_SEQUENCE_ENCODER_TYPE', 'mamba')
    sequence_embedding_dim = int(os.environ.get('CARBONBENCH_SEQUENCE_EMBEDDING_DIM', '128'))
    model_dropout = float(os.environ.get('CARBONBENCH_MODEL_DROPOUT', '0.1'))
    mamba_d_model = int(os.environ.get('CARBONBENCH_MAMBA_D_MODEL', str(sequence_embedding_dim)))
    mamba_layers = int(os.environ.get('CARBONBENCH_MAMBA_LAYERS', '2'))
    mamba_d_state = int(os.environ.get('CARBONBENCH_MAMBA_D_STATE', '16'))
    mamba_d_conv = int(os.environ.get('CARBONBENCH_MAMBA_D_CONV', '4'))
    mamba_expand = int(os.environ.get('CARBONBENCH_MAMBA_EXPAND', '2'))
    transformer_nhead = int(os.environ.get('CARBONBENCH_TRANSFORMER_NHEAD', '4'))
    transformer_dim_feedforward_env = os.environ.get('CARBONBENCH_TRANSFORMER_DIM_FEEDFORWARD', '')
    transformer_dim_feedforward = int(transformer_dim_feedforward_env) if transformer_dim_feedforward_env else None
    experiment_name = os.environ.get(
        'CARBONBENCH_EXPERIMENT_NAME',
        (
            'carbonbench_flux_gpp_cnn_mamba_smoke'
            if use_image_branch and use_sequence_branch
            else 'carbonbench_flux_gpp_image_smoke'
            if use_image_branch
            else 'carbonbench_flux_gpp_baseline'
        )
    )

    fixed_train_sites_env = os.environ.get('CARBONBENCH_FIXED_TRAIN_SITES', '')
    fixed_val_sites_env = os.environ.get('CARBONBENCH_FIXED_VAL_SITES', '')
    fixed_train_sites = [s.strip() for s in fixed_train_sites_env.split(',') if s.strip()] if fixed_train_sites_env else None
    fixed_val_sites = [s.strip() for s in fixed_val_sites_env.split(',') if s.strip()] if fixed_val_sites_env else None
    target_columns_env = os.environ.get('CARBONBENCH_TARGET_COLUMNS', '').strip()
    target_columns = [s.strip() for s in target_columns_env.split(',') if s.strip()] if target_columns_env else None
    target_column = os.environ.get('CARBONBENCH_TARGET_COLUMN', 'GPP_NT_VUT_USTAR50')
    output_dim = len(target_columns) if target_columns else 1

    data_seed = int(os.environ.get('CARBONBENCH_DATA_SEED', os.environ.get('CARBONBENCH_SEED', '42')))
    model_seed = int(os.environ.get('CARBONBENCH_MODEL_SEED', os.environ.get('CARBONBENCH_SEED', '42')))

    config['data'] = {
        'dataset_type': 'carbonbench_flux',
        'data_root': data_root,
        'target_flux_file': os.path.join(data_root, 'target_fluxes.parquet'),
        'modis_file': os.path.join(data_root, 'MOD09GA.parquet'),
        'era5_file': os.environ.get('CARBONBENCH_ERA5_FILE', os.path.join(data_root, 'ERA5.parquet')),
        'feature_sets_file': os.path.join(data_root, 'feature_sets.json'),
        'koppen_sites_file': os.path.join(data_root, 'koppen_sites.json'),
        'split_file': split_file,
        'split_mode': 'source_val_target',
        'source_val_fraction_sites': 0.2,
        'target_column': target_columns if target_columns else target_column,
        'feature_set_name': feature_set_name,
        'extra_feature_columns': [
            'sur_refl_b01',
            'sur_refl_b02',
            'sur_refl_b03',
            'sur_refl_b04',
            'sur_refl_b05',
            'sur_refl_b06',
            'sur_refl_b07',
            'SensorZenith',
            'SensorAzimuth',
            'SolarZenith',
            'SolarAzimuth',
            'clouds',
        ],
        'batch_size': 256,
        'val_batch_size': 512,
        'test_batch_size': 512,
        'num_workers': 0,
        'seed': data_seed,
        'min_observation_days': 1095,
        'use_image_branch': use_image_branch,
        'include_patch_fields': include_patch_fields,
        'patch_size_km': 2.0,
        'patch_manifest_file': patch_manifest_file,
        'restrict_to_patch_manifest_rows': restrict_to_patch_manifest_rows,
        'restrict_to_patch_manifest_sites': restrict_to_patch_manifest_sites,
        'image_context_mode': image_context_mode,
        'image_context_max_patches': image_context_max_patches,
        'image_channels': 6,
        'use_sequence_branch': use_sequence_branch,
        'igbp_filter': igbp_filter,
        'sequence_length': sequence_length,
        'sequence_include_current': sequence_include_current,
        'sequence_encoder_type': sequence_encoder_type,
        'sequence_stride': int(os.environ.get('CARBONBENCH_SEQUENCE_STRIDE', '1')),
        'sequence_extra_feature_columns': ['doy_sin', 'doy_cos'],
        'standardize_targets': os.environ.get('CARBONBENCH_STANDARDIZE_TARGETS', '0') == '1',
        'eval_qc_threshold': (
            float(os.environ['CARBONBENCH_EVAL_QC_THRESHOLD'])
            if os.environ.get('CARBONBENCH_EVAL_QC_THRESHOLD', '').strip()
            else None
        ),
        'use_carbonbench_sample_weights': os.environ.get('CARBONBENCH_USE_SAMPLE_WEIGHTS', '0') == '1',
        'max_train_rows': None,
        'max_val_rows': None,
        'max_test_rows': None,
        'fixed_train_sites': fixed_train_sites,
        'fixed_val_sites': fixed_val_sites,
    }

    config['model'] = {
        'name': 'FluxTransferModel',
        'params': {
            'tabular_input_dim': 0,
            'metadata_input_dim': 0,
            'tabular_hidden_dims': [256, 128],
            'metadata_hidden_dims': [32],
            'fusion_hidden_dims': [128, 64],
            'dropout': model_dropout,
            'use_image_branch': use_image_branch,
            'image_patch_dim': 0,
            'image_channels': 6,
            'image_encoder_type': image_encoder_type,
            'image_resnet_variant': image_resnet_variant,
            'image_resnet_pretrained': image_resnet_pretrained,
            'image_branch_trainable': image_branch_trainable,
            'image_branch_hidden_dims': [32, 64, 128],
            'image_embedding_dim': 128,
            'use_sequence_branch': use_sequence_branch,
            'sequence_encoder_type': sequence_encoder_type,
            'sequence_input_dim': 0,
            'sequence_embedding_dim': sequence_embedding_dim,
            'mamba_d_model': mamba_d_model,
            'mamba_layers': mamba_layers,
            'mamba_d_state': mamba_d_state,
            'mamba_d_conv': mamba_d_conv,
            'mamba_expand': mamba_expand,
            'transformer_nhead': transformer_nhead,
            'transformer_dim_feedforward': transformer_dim_feedforward,
            'fusion_mode': fusion_mode,
            'image_gate_init_bias': image_gate_init_bias,
            'output_dim': output_dim,
        }
    }

    criterion_name = os.environ.get('CARBONBENCH_CRITERION', 'HuberLoss')
    config['training'] = {
        'task_type': 'regression',
        'epochs': 20,
        'optimizer': 'AdamW',
        'optimizer_params': {'lr': 1e-3, 'weight_decay': 1e-4},
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_params': {'mode': 'min', 'factor': 0.5, 'patience': 2},
        'criterion': criterion_name,
        'criterion_params': {} if criterion_name == 'CarbonBenchFluxLoss' else {'delta': 1.0},
        'flux_constraint_weight': float(os.environ.get('CARBONBENCH_FLUX_CONSTRAINT_WEIGHT', '0.0')),
        'aux_loss_weight': 0.0,
        'clip_grad_norm': 1.0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dtype': 'float32',
        'seed': model_seed,
        'eval_freq': 1,
        'print_freq': 50,
        'checkpoint_dir': os.environ.get('CARBONBENCH_CHECKPOINT_DIR', './checkpoints'),
        'experiment_name': experiment_name,
        'max_recent_checkpoints_to_keep': 3,
        'primary_metric_name': os.environ.get('CARBONBENCH_PRIMARY_METRIC_NAME', 'rmse'),
        'primary_metric_mode': os.environ.get('CARBONBENCH_PRIMARY_METRIC_MODE', 'min'),
    }

    return config


def get_carbonbench_flux_hiermoe_config():
    """
    Config for FluxHierarchicalMoEModel:
      - Lightweight multi-scale ResNet18 image branch with temporal FiLM conditioning
      - Dense 30-day temporal Mamba/GRU sequence branch
      - MoE regression head (shared + 4 routed experts, top-k=2)

    Key env vars (inherits all from base config, plus new ones below):
      CARBONBENCH_ADAPTER_DIM        default 128
      CARBONBENCH_MOE_HIDDEN_DIM     default 256
      CARBONBENCH_MOE_N_EXPERTS      default 4
      CARBONBENCH_MOE_TOP_K          default 2
      CARBONBENCH_MOE_AUX_ALPHA      default 1.0
      CARBONBENCH_AUX_LOSS_WEIGHT    default 0.01
    """
    # Start from the base config (data + training sections are identical)
    cfg = get_carbonbench_flux_from_scratch_config()

    # Read new hyper-params
    adapter_dim       = int(os.environ.get('CARBONBENCH_ADAPTER_DIM',    '128'))
    moe_hidden_dim    = int(os.environ.get('CARBONBENCH_MOE_HIDDEN_DIM', '256'))
    moe_n_experts     = int(os.environ.get('CARBONBENCH_MOE_N_EXPERTS',  '4'))
    moe_top_k         = int(os.environ.get('CARBONBENCH_MOE_TOP_K',      '2'))
    moe_aux_alpha     = float(os.environ.get('CARBONBENCH_MOE_AUX_ALPHA', '1.0'))
    aux_loss_weight   = float(os.environ.get('CARBONBENCH_AUX_LOSS_WEIGHT', '0.01'))
    use_moe_head      = os.environ.get('CARBONBENCH_USE_MOE_HEAD', '1') == '1'
    target_columns_env = os.environ.get('CARBONBENCH_TARGET_COLUMNS', '').strip()
    target_columns = [s.strip() for s in target_columns_env.split(',') if s.strip()] if target_columns_env else None
    output_dim = len(target_columns) if target_columns else 1

    # Read vars already parsed by the base config
    use_image_branch    = os.environ.get('CARBONBENCH_USE_IMAGE_BRANCH',    '0') == '1'
    use_sequence_branch = os.environ.get('CARBONBENCH_USE_SEQUENCE_BRANCH', '0') == '1'
    sequence_encoder_type  = os.environ.get('CARBONBENCH_SEQUENCE_ENCODER_TYPE', 'mamba')
    sequence_embedding_dim = int(os.environ.get('CARBONBENCH_SEQUENCE_EMBEDDING_DIM', '128'))
    model_dropout = float(os.environ.get('CARBONBENCH_MODEL_DROPOUT', '0.1'))
    mamba_d_model = int(os.environ.get('CARBONBENCH_MAMBA_D_MODEL', str(sequence_embedding_dim)))
    mamba_layers  = int(os.environ.get('CARBONBENCH_MAMBA_LAYERS',  '2'))
    mamba_d_state = int(os.environ.get('CARBONBENCH_MAMBA_D_STATE', '16'))
    mamba_d_conv  = int(os.environ.get('CARBONBENCH_MAMBA_D_CONV',  '4'))
    mamba_expand  = int(os.environ.get('CARBONBENCH_MAMBA_EXPAND',  '2'))
    transformer_nhead = int(os.environ.get('CARBONBENCH_TRANSFORMER_NHEAD', '4'))
    transformer_dim_feedforward_env = os.environ.get('CARBONBENCH_TRANSFORMER_DIM_FEEDFORWARD', '')
    transformer_dim_feedforward = int(transformer_dim_feedforward_env) if transformer_dim_feedforward_env else None
    image_resnet_pretrained = os.environ.get('CARBONBENCH_IMAGE_RESNET_PRETRAINED', '0') == '1'

    cfg['model'] = {
        'name': 'FluxHierarchicalMoEModel',
        'params': {
            # tabular / metadata (dims filled by dataloader at runtime, kept as 0 here)
            'tabular_input_dim' : 0,
            'metadata_input_dim': 0,
            'tabular_hidden_dims' : [256, 128],
            'metadata_hidden_dims': [32],
            'dropout': model_dropout,
            # temporal branch
            'use_sequence_branch'  : use_sequence_branch,
            'sequence_input_dim'   : 0,
            'sequence_encoder_type': sequence_encoder_type,
            'sequence_embedding_dim': sequence_embedding_dim,
            'mamba_d_model': mamba_d_model,
            'mamba_layers' : mamba_layers,
            'mamba_d_state': mamba_d_state,
            'mamba_d_conv' : mamba_d_conv,
            'mamba_expand' : mamba_expand,
            'transformer_nhead': transformer_nhead,
            'transformer_dim_feedforward': transformer_dim_feedforward,
            # image branch
            'use_image_branch'      : use_image_branch,
            'image_channels'        : 6,
            'image_resnet_pretrained': image_resnet_pretrained,
            'adapter_dim'           : adapter_dim,
            # MoE head
            'moe_hidden_dim'   : moe_hidden_dim,
            'moe_n_experts'    : moe_n_experts,
            'moe_top_k'        : moe_top_k,
            'moe_aux_loss_alpha': moe_aux_alpha,
            'use_moe_head'     : use_moe_head,
            'output_dim'       : output_dim,
        }
    }

    # Scale the MoE load-balancing loss in the training loop. The MoE head
    # returns an unweighted loss by default, so avoid setting both knobs to 0.01.
    cfg['training']['aux_loss_weight'] = aux_loss_weight

    return cfg
