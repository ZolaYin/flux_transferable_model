import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


CARBONBENCH_FOREST_CLASSES = {'ENF', 'EBF', 'DNF', 'DBF', 'MF'}
TIME_COLUMN_CANDIDATES = ['date', 'TIMESTAMP', 'timestamp']


class CarbonBenchFluxDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tabular_columns: List[str],
        metadata_columns: List[str],
        target_column: str,
        include_patch_fields: bool = True,
        sequence_columns: List[str] = None,
        sequence_length: int = 0,
        sequence_include_current: bool = True,
        sequence_frame: Optional[pd.DataFrame] = None,
        sample_weight_column: Optional[str] = None,
        max_context_patches: int = 1,
        patch_shape: Tuple[int, int, int] = (6, 67, 67),
    ):
        self.frame = frame.reset_index(drop=True)
        self.tabular_columns = tabular_columns
        self.metadata_columns = metadata_columns
        self.target_column = target_column
        self.target_columns = target_column if isinstance(target_column, list) else None
        self.include_patch_fields = include_patch_fields
        self.sequence_columns = sequence_columns or []
        self.sequence_length = int(sequence_length)
        self.sequence_include_current = bool(sequence_include_current)
        self.sample_weight_column = sample_weight_column
        self.max_context_patches = max(1, int(max_context_patches))
        self.patch_shape = tuple(int(v) for v in patch_shape)
        self.sequence_enabled = bool(self.sequence_columns) and self.sequence_length > 0
        self.sequence_row_indices: Dict[int, List[int]] = {}
        self._dense_sequences: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None
        self._dense_sequence_values: Optional[np.ndarray] = None
        self._dense_sequence_masks: Optional[np.ndarray] = None
        if self.sequence_enabled:
            if sequence_frame is not None:
                self._precompute_dense_sequences(sequence_frame)
            else:
                self.sequence_row_indices = self._build_sequence_row_indices()

    def __len__(self):
        return len(self.frame)

    def _precompute_dense_sequences(self, sequence_frame: pd.DataFrame) -> None:
        col_positions = {c: idx for idx, c in enumerate(self.sequence_columns)}
        site_frames: Dict[str, Tuple[pd.DataFrame, pd.DatetimeIndex]] = {}
        for site, sdf in sequence_frame.groupby('site', sort=False):
            ordered = sdf.sort_values('date').reset_index(drop=True).copy()
            ordered['date'] = pd.to_datetime(ordered['date'], errors='coerce').dt.normalize()
            site_frames[site] = (ordered, pd.DatetimeIndex(ordered['date']))

        seq_len = self.sequence_length
        nrows = len(self.frame)
        ncols = len(self.sequence_columns)
        sequence_values = np.zeros((nrows, seq_len, ncols), dtype=np.float32)
        sequence_masks = np.zeros((nrows, seq_len), dtype=np.float32)
        side = 'right' if self.sequence_include_current else 'left'

        for site, frame_site in self.frame.groupby('site', sort=False):
            site_payload = site_frames.get(site)
            if site_payload is None or len(site_payload[0]) == 0:
                continue
            sdf, date_index = site_payload
            valid_cols = [c for c in self.sequence_columns if c in sdf.columns]
            if not valid_cols:
                continue

            site_values = np.zeros((len(sdf), ncols), dtype=np.float32)
            vals = sdf[valid_cols].to_numpy(dtype=np.float32)
            np.nan_to_num(vals, copy=False)
            col_idx = [col_positions[c] for c in valid_cols]
            site_values[:, col_idx] = vals

            row_indices = frame_site.index.to_numpy(dtype=np.int64)
            query_dates = pd.to_datetime(frame_site['date'], errors='coerce').dt.normalize()
            valid_date_mask = query_dates.notna().to_numpy()
            if not valid_date_mask.any():
                continue

            valid_rows = row_indices[valid_date_mask]
            positions = date_index.searchsorted(pd.DatetimeIndex(query_dates[valid_date_mask]), side=side)
            for step in range(seq_len):
                src_idx = positions - seq_len + step
                valid_src = (src_idx >= 0) & (src_idx < len(site_values))
                if not valid_src.any():
                    continue
                sequence_values[valid_rows[valid_src], step, :] = site_values[src_idx[valid_src]]
                sequence_masks[valid_rows[valid_src], step] = 1.0

        self._dense_sequence_values = sequence_values
        self._dense_sequence_masks = sequence_masks

    def _build_sequence_row_indices(self) -> Dict[int, List[int]]:
        sequence_map: Dict[int, List[int]] = {}
        grouped = self.frame.groupby('site', sort=False)
        for _, site_frame in grouped:
            ordered_indices = site_frame.sort_values(['date', 'TIMESTAMP']).index.to_list()
            for pos, row_index in enumerate(ordered_indices):
                end_pos = pos + 1 if self.sequence_include_current else pos
                start_pos = max(0, end_pos - self.sequence_length)
                sequence_map[row_index] = ordered_indices[start_pos:end_pos]
        return sequence_map

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = row.get('image_path', '')
        image_paths_value = row.get('image_paths', image_path)
        batch = {
            'tabular_features': torch.tensor(row[self.tabular_columns].to_numpy(dtype=np.float32), dtype=torch.float32),
            'metadata_features': torch.tensor(row[self.metadata_columns].to_numpy(dtype=np.float32), dtype=torch.float32),
            'site_id': row['site'],
            'timestamp': int(row['TIMESTAMP']),
            'image_path': image_path,
        }
        if self.sample_weight_column and self.sample_weight_column in row.index:
            batch['sample_weight'] = torch.tensor(float(row[self.sample_weight_column]), dtype=torch.float32)
        if self.sequence_enabled:
            if self._dense_sequence_values is not None and self._dense_sequence_masks is not None:
                sequence_array = self._dense_sequence_values[index]
                sequence_mask = self._dense_sequence_masks[index]
            else:
                sequence_indices = self.sequence_row_indices.get(index, [])
                sequence_array = np.zeros((self.sequence_length, len(self.sequence_columns)), dtype=np.float32)
                sequence_mask = np.zeros((self.sequence_length,), dtype=np.float32)
                if sequence_indices:
                    sequence_values = self.frame.loc[sequence_indices, self.sequence_columns].to_numpy(dtype=np.float32)
                    valid_length = min(len(sequence_values), self.sequence_length)
                    sequence_array[-valid_length:] = sequence_values[-valid_length:]
                    sequence_mask[-valid_length:] = 1.0
            batch['sequence_features'] = torch.tensor(sequence_array, dtype=torch.float32)
            batch['sequence_mask'] = torch.tensor(sequence_mask, dtype=torch.float32)
        if self.include_patch_fields:
            image_paths = [
                part for part in str(image_paths_value).split('||')
                if part and part.strip()
            ][:self.max_context_patches]
            if image_paths:
                patch_arrays = []
                for path in image_paths:
                    with np.load(path) as patch_data:
                        patch_array = patch_data['image'].astype(np.float32)
                    patch_array = np.nan_to_num(patch_array, nan=0.0, posinf=0.0, neginf=0.0)
                    if np.nanmax(np.abs(patch_array)) > 10.0:
                        patch_array = patch_array / 10000.0
                    patch_arrays.append(patch_array)

                if self.max_context_patches > 1:
                    patch_shape = patch_arrays[0].shape if patch_arrays else self.patch_shape
                    padded = np.zeros((self.max_context_patches, *patch_shape), dtype=np.float32)
                    mask = np.zeros((self.max_context_patches,), dtype=np.float32)
                    for patch_idx, patch_array in enumerate(patch_arrays[:self.max_context_patches]):
                        padded[patch_idx] = patch_array
                        mask[patch_idx] = 1.0
                    batch['patch_tensor'] = torch.tensor(padded, dtype=torch.float32)
                    batch['patch_mask'] = torch.tensor(mask, dtype=torch.float32)
                else:
                    batch['patch_tensor'] = torch.tensor(patch_arrays[0], dtype=torch.float32)
            else:
                batch['patch_tensor'] = torch.empty(0, dtype=torch.float32)
                if self.max_context_patches > 1:
                    batch['patch_mask'] = torch.zeros((self.max_context_patches,), dtype=torch.float32)
        if self.target_columns:
            target_values = row[self.target_columns].to_numpy(dtype=np.float32)
            target = torch.tensor(target_values, dtype=torch.float32)
        else:
            target = torch.tensor(float(row[self.target_column]), dtype=torch.float32)
        return batch, target


def _detect_time_column(schema: List[str]) -> str:
    schema_set = set(schema)
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in schema_set:
            return candidate
    raise ValueError(f"Could not find a supported time column in schema. Expected one of {TIME_COLUMN_CANDIDATES}, got {schema}")


def _read_schema_columns(path: Path) -> List[str]:
    return pq.ParquetFile(path).schema.names


def _load_parquet_subset(path: Path, columns: List[str], site_ids: List[str]) -> pd.DataFrame:
    filters = [('site', 'in', site_ids)]
    return pd.read_parquet(path, columns=columns, filters=filters)


def _normalize_timestamp_columns(frame: pd.DataFrame, time_column: str) -> pd.DataFrame:
    frame = frame.copy()
    if time_column == 'date':
        normalized_date = pd.to_datetime(frame['date'], errors='coerce').dt.normalize()
        frame['date'] = normalized_date
        if 'TIMESTAMP' not in frame.columns:
            frame['TIMESTAMP'] = normalized_date.dt.strftime('%Y%m%d')
    else:
        timestamp_series = frame[time_column].astype(str)
        normalized_date = pd.to_datetime(timestamp_series, format='%Y%m%d', errors='coerce').dt.normalize()
        frame['date'] = normalized_date
        if time_column != 'TIMESTAMP':
            frame['TIMESTAMP'] = timestamp_series

    frame['TIMESTAMP'] = pd.to_numeric(frame['TIMESTAMP'], errors='coerce').astype('Int64')
    return frame


def _one_hot_encode(frame: pd.DataFrame, categorical_cols: List[str], categories_by_col: Dict[str, List[str]]) -> pd.DataFrame:
    encoded_parts = []
    for col in categorical_cols:
        categories = categories_by_col[col]
        data = np.zeros((len(frame), len(categories)), dtype=np.float32)
        col_values = frame[col].fillna('UNK').to_numpy()
        index_map = {category: idx for idx, category in enumerate(categories)}
        for row_idx, value in enumerate(col_values):
            if value in index_map:
                data[row_idx, index_map[value]] = 1.0
        col_names = [f'{col}__{category}' for category in categories]
        encoded_parts.append(pd.DataFrame(data, columns=col_names, index=frame.index))
    return pd.concat(encoded_parts, axis=1) if encoded_parts else pd.DataFrame(index=frame.index)


def _standardize_split(
    train_df: pd.DataFrame,
    other_frames: List[pd.DataFrame],
    columns: List[str],
) -> Tuple[pd.DataFrame, List[pd.DataFrame], pd.Series, pd.Series, pd.Series]:
    fill_values = train_df[columns].median(numeric_only=True)
    train_df.loc[:, columns] = train_df[columns].fillna(fill_values)
    train_mean = train_df[columns].mean()
    train_std = train_df[columns].std().replace(0, 1.0).fillna(1.0)
    train_df.loc[:, columns] = (train_df[columns] - train_mean) / train_std

    transformed_frames = []
    for frame in other_frames:
        frame = frame.copy()
        frame.loc[:, columns] = frame[columns].fillna(fill_values)
        frame.loc[:, columns] = (frame[columns] - train_mean) / train_std
        transformed_frames.append(frame)
    return train_df, transformed_frames, train_mean, train_std, fill_values


def _prepare_metadata(frame: pd.DataFrame, koppen_sites_path: Path) -> pd.DataFrame:
    koppen_sites = json.load(open(koppen_sites_path))
    frame = frame.copy()
    frame['koppen_main'] = frame['site'].map(koppen_sites).fillna('UNK')
    timestamps = pd.to_datetime(frame['TIMESTAMP'].astype(str), format='%Y%m%d', errors='coerce')
    day_of_year = timestamps.dt.dayofyear.fillna(1).astype(np.int32)
    frame['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365.25).astype(np.float32)
    frame['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365.25).astype(np.float32)
    frame['image_path'] = ''
    return frame


def _merge_patch_manifest(
    frame: pd.DataFrame,
    patch_manifest_path: Path,
    data_root: Path,
    use_image_branch: bool,
    restrict_to_manifest_rows: bool = False,
    image_context_mode: str = 'exact',
    max_context_patches: int = 1,
) -> pd.DataFrame:
    if not patch_manifest_path.exists():
        return frame

    patch_df = pd.read_csv(patch_manifest_path)
    if patch_df.empty:
        return frame

    patch_df = patch_df.copy()
    patch_df['date'] = pd.to_datetime(patch_df['date'], errors='coerce').dt.normalize()
    patch_df['site'] = patch_df['site_id']
    patch_df['patch_path'] = patch_df['patch_path'].fillna('').astype(str)

    def resolve_patch_path(path_value: str) -> str:
        if not path_value:
            return ''
        path = Path(path_value)
        if path.is_absolute():
            return str(path)
        candidates = [
            (data_root / path),
            (data_root.parent / path),
            (data_root.parent.parent / path),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        return str((data_root.parent.parent / path).resolve())

    patch_df['patch_path'] = patch_df['patch_path'].apply(resolve_patch_path)
    patch_df['patch_exists'] = patch_df['patch_path'].apply(lambda p: bool(p) and Path(p).exists())
    if 'download_status' in patch_df.columns:
        valid_statuses = {'ok', 'exists'}
        patch_df = patch_df[
            patch_df['download_status'].fillna('').astype(str).str.lower().isin(valid_statuses)
            & patch_df['patch_exists']
        ].copy()
    else:
        patch_df = patch_df[patch_df['patch_exists']].copy()
    keep_columns = [
        'site', 'date', 'patch_path', 'image_date', 'cloud_qa', 'sensor',
        'collection', 'granule_id', 'download_status'
    ]
    keep_columns = [col for col in keep_columns if col in patch_df.columns]
    patch_df = patch_df[keep_columns].copy()
    if 'cloud_qa' in patch_df.columns:
        patch_df['_cloud_sort'] = pd.to_numeric(patch_df['cloud_qa'], errors='coerce').fillna(np.inf)
    else:
        patch_df['_cloud_sort'] = np.inf
    patch_df['_date_sort'] = pd.to_datetime(patch_df['image_date'].fillna(patch_df['date']), errors='coerce')
    patch_df = patch_df.sort_values(['site', '_cloud_sort', '_date_sort'])

    image_context_mode = str(image_context_mode or 'exact').strip().lower()
    max_context_patches = max(1, int(max_context_patches))

    def add_path_list(frame_in: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        grouped_paths = (
            frame_in.groupby(group_cols, sort=False)['patch_path']
            .apply(lambda paths: '||'.join(paths.head(max_context_patches).astype(str).tolist()))
            .reset_index()
            .rename(columns={'patch_path': 'image_paths'})
        )
        first_rows = frame_in.drop_duplicates(subset=group_cols, keep='first')
        context = first_rows.merge(grouped_paths, on=group_cols, how='left')
        return context

    if image_context_mode in {'site_pool', 'site_multi', 'site_mean'}:
        context_df = add_path_list(patch_df, ['site'])
        merged = frame.merge(context_df, on='site', how='left', suffixes=('', '_patch'))
    elif image_context_mode in {'site', 'site_context', 'static'}:
        context_df = add_path_list(patch_df, ['site'])
        context_df['image_paths'] = context_df['patch_path']
        merged = frame.merge(context_df, on='site', how='left', suffixes=('', '_patch'))
    elif image_context_mode in {'month_pool', 'monthly_pool', 'month_multi'}:
        patch_df['context_month'] = patch_df['date'].dt.month
        context_df = add_path_list(patch_df, ['site', 'context_month'])
        merged = frame.copy()
        merged['context_month'] = pd.to_datetime(merged['date'], errors='coerce').dt.month
        merged = merged.merge(context_df, on=['site', 'context_month'], how='left', suffixes=('', '_patch'))
        fallback_df = add_path_list(patch_df, ['site'])[['site', 'patch_path', 'image_paths']]
        fallback_df = fallback_df.rename(columns={'patch_path': 'fallback_patch_path', 'image_paths': 'fallback_image_paths'})
        merged = merged.merge(fallback_df, on='site', how='left')
        merged['patch_path'] = merged['patch_path'].fillna(merged['fallback_patch_path'])
        merged['image_paths'] = merged['image_paths'].fillna(merged['fallback_image_paths'])
        merged = merged.drop(columns=[col for col in ['fallback_patch_path', 'fallback_image_paths'] if col in merged.columns])
    elif image_context_mode in {'month', 'monthly'}:
        patch_df['context_month'] = patch_df['date'].dt.month
        context_df = add_path_list(patch_df, ['site', 'context_month'])
        context_df['image_paths'] = context_df['patch_path']
        merged = frame.copy()
        merged['context_month'] = pd.to_datetime(merged['date'], errors='coerce').dt.month
        merged = merged.merge(context_df, on=['site', 'context_month'], how='left', suffixes=('', '_patch'))
        fallback_df = add_path_list(patch_df, ['site'])[['site', 'patch_path', 'image_paths']]
        fallback_df = fallback_df.rename(columns={'patch_path': 'fallback_patch_path', 'image_paths': 'fallback_image_paths'})
        merged = merged.merge(fallback_df, on='site', how='left')
        merged['patch_path'] = merged['patch_path'].fillna(merged['fallback_patch_path'])
        merged['image_paths'] = merged['image_paths'].fillna(merged['fallback_image_paths'])
        merged = merged.drop(columns=[col for col in ['fallback_patch_path', 'fallback_image_paths'] if col in merged.columns])
    elif image_context_mode in {'season_pool', 'seasonal_pool', 'season_multi'}:
        def season_from_month(month_series):
            return ((month_series.fillna(1).astype(int) % 12) // 3).astype(int)

        patch_df['context_season'] = season_from_month(patch_df['date'].dt.month)
        context_df = add_path_list(patch_df, ['site', 'context_season'])
        merged = frame.copy()
        merged['context_season'] = season_from_month(pd.to_datetime(merged['date'], errors='coerce').dt.month)
        merged = merged.merge(context_df, on=['site', 'context_season'], how='left', suffixes=('', '_patch'))
        fallback_df = add_path_list(patch_df, ['site'])[['site', 'patch_path', 'image_paths']]
        fallback_df = fallback_df.rename(columns={'patch_path': 'fallback_patch_path', 'image_paths': 'fallback_image_paths'})
        merged = merged.merge(fallback_df, on='site', how='left')
        merged['patch_path'] = merged['patch_path'].fillna(merged['fallback_patch_path'])
        merged['image_paths'] = merged['image_paths'].fillna(merged['fallback_image_paths'])
        merged = merged.drop(columns=[col for col in ['fallback_patch_path', 'fallback_image_paths'] if col in merged.columns])
    elif image_context_mode in {'season', 'seasonal'}:
        def season_from_month(month_series):
            return ((month_series.fillna(1).astype(int) % 12) // 3).astype(int)

        patch_df['context_season'] = season_from_month(patch_df['date'].dt.month)
        context_df = add_path_list(patch_df, ['site', 'context_season'])
        context_df['image_paths'] = context_df['patch_path']
        merged = frame.copy()
        merged['context_season'] = season_from_month(pd.to_datetime(merged['date'], errors='coerce').dt.month)
        merged = merged.merge(context_df, on=['site', 'context_season'], how='left', suffixes=('', '_patch'))
        fallback_df = add_path_list(patch_df, ['site'])[['site', 'patch_path', 'image_paths']]
        fallback_df = fallback_df.rename(columns={'patch_path': 'fallback_patch_path', 'image_paths': 'fallback_image_paths'})
        merged = merged.merge(fallback_df, on='site', how='left')
        merged['patch_path'] = merged['patch_path'].fillna(merged['fallback_patch_path'])
        merged['image_paths'] = merged['image_paths'].fillna(merged['fallback_image_paths'])
        merged = merged.drop(columns=[col for col in ['fallback_patch_path', 'fallback_image_paths'] if col in merged.columns])
    else:
        context_df = patch_df.drop_duplicates(subset=['site', 'date'], keep='first')
        context_df['image_paths'] = context_df['patch_path']
        merged = frame.merge(context_df, on=['site', 'date'], how='left')

    merged['image_path'] = merged['patch_path'].fillna('').astype(str)
    if 'image_paths' not in merged.columns:
        merged['image_paths'] = merged['image_path']
    merged['image_paths'] = merged['image_paths'].fillna(merged['image_path']).astype(str)
    if use_image_branch or restrict_to_manifest_rows:
        merged = merged[merged['image_path'].str.len() > 0].copy()
    return merged


def _valid_patch_manifest_sites(patch_manifest_path: Path, data_root: Path) -> set:
    if not patch_manifest_path or not patch_manifest_path.exists():
        return set()

    patch_df = pd.read_csv(patch_manifest_path)
    if patch_df.empty or 'site_id' not in patch_df.columns:
        return set()

    patch_df = patch_df.copy()
    if 'patch_path' not in patch_df.columns:
        return set(patch_df['site_id'].dropna().astype(str).tolist())

    patch_df['patch_path'] = patch_df['patch_path'].fillna('').astype(str)

    def resolve_patch_path(path_value: str) -> str:
        if not path_value:
            return ''
        path = Path(path_value)
        if path.is_absolute():
            return str(path)
        candidates = [
            (data_root / path),
            (data_root.parent / path),
            (data_root.parent.parent / path),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        return str((data_root.parent.parent / path).resolve())

    patch_df['patch_path'] = patch_df['patch_path'].apply(resolve_patch_path)
    patch_df['patch_exists'] = patch_df['patch_path'].apply(lambda p: bool(p) and Path(p).exists())
    if 'download_status' in patch_df.columns:
        valid_statuses = {'ok', 'exists'}
        patch_df = patch_df[
            patch_df['download_status'].fillna('').astype(str).str.lower().isin(valid_statuses)
            & patch_df['patch_exists']
        ].copy()
    else:
        patch_df = patch_df[patch_df['patch_exists']].copy()

    return set(patch_df['site_id'].dropna().astype(str).tolist())


def _resolve_igbp_filter(raw_filter) -> Optional[set]:
    if raw_filter is None:
        return CARBONBENCH_FOREST_CLASSES
    if isinstance(raw_filter, (list, tuple, set)):
        return {str(item).strip() for item in raw_filter if str(item).strip()}

    value = str(raw_filter).strip()
    if not value or value.lower() in {'all', '*', 'none'}:
        return None
    if value.lower() == 'forest':
        return CARBONBENCH_FOREST_CLASSES
    return {item.strip() for item in value.split(',') if item.strip()}


def _apply_temporal_stride(frame: pd.DataFrame, stride: int, offset: int = 0) -> pd.DataFrame:
    if stride <= 1 or frame.empty:
        return frame
    selected = []
    for _, site_frame in frame.sort_values(['site', 'date', 'TIMESTAMP']).groupby('site', sort=False):
        positions = np.arange(len(site_frame))
        selected.append(site_frame.iloc[(positions - offset) % stride == 0])
    if not selected:
        return frame.iloc[0:0].copy()
    return pd.concat(selected, axis=0).sort_index().copy()


def _add_sample_weights(
    train_df: pd.DataFrame,
    other_frames: List[pd.DataFrame],
    qc_column: str = 'NEE_VUT_USTAR50_QC',
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    train_df = train_df.copy()
    other_frames = [frame.copy() for frame in other_frames]
    igbp_counts = train_df['IGBP'].fillna('UNK').value_counts()
    koppen_counts = train_df['koppen_main'].fillna('UNK').value_counts()
    igbp_weights = (len(train_df) / (len(igbp_counts) * igbp_counts)).to_dict()
    koppen_weights = (len(train_df) / (len(koppen_counts) * koppen_counts)).to_dict()

    def apply_weights(frame: pd.DataFrame) -> pd.DataFrame:
        qc = frame[qc_column].fillna(1.0).astype(float).clip(lower=0.0, upper=1.0) if qc_column in frame.columns else 1.0
        wi = frame['IGBP'].fillna('UNK').map(igbp_weights).fillna(1.0).astype(float)
        wk = frame['koppen_main'].fillna('UNK').map(koppen_weights).fillna(1.0).astype(float)
        weight = np.asarray(qc, dtype=np.float64) * wi.to_numpy(dtype=np.float64) * wk.to_numpy(dtype=np.float64)
        finite = np.isfinite(weight)
        if finite.any() and weight[finite].mean() > 0:
            weight = weight / weight[finite].mean()
        frame['sample_weight'] = weight.astype(np.float32)
        return frame

    return apply_weights(train_df), [apply_weights(frame) for frame in other_frames]


def get_carbonbench_flux_dataloaders(config: dict):
    data_cfg = config['data']
    data_root = Path(data_cfg['data_root'])
    target_path = Path(data_cfg['target_flux_file'])
    modis_path = Path(data_cfg['modis_file'])
    era5_path = Path(data_cfg['era5_file'])
    feature_sets_path = Path(data_cfg['feature_sets_file'])
    koppen_sites_path = Path(data_cfg['koppen_sites_file'])
    split_path = Path(data_cfg['split_file'])
    patch_manifest_path_str = data_cfg.get('patch_manifest_file')
    patch_manifest_path = Path(patch_manifest_path_str) if patch_manifest_path_str else None
    restrict_to_manifest_rows = bool(data_cfg.get('restrict_to_patch_manifest_rows', False))
    restrict_to_manifest_sites = bool(data_cfg.get('restrict_to_patch_manifest_sites', False))
    image_context_mode = str(data_cfg.get('image_context_mode', 'exact') or 'exact')
    image_context_max_patches = int(data_cfg.get('image_context_max_patches', 1) or 1)

    split_df = pd.read_csv(split_path)
    if restrict_to_manifest_sites:
        if not patch_manifest_path:
            raise ValueError("restrict_to_patch_manifest_sites requires patch_manifest_file.")
        manifest_sites = _valid_patch_manifest_sites(patch_manifest_path, data_root)
        if not manifest_sites:
            raise ValueError(f"No valid patch sites found in manifest: {patch_manifest_path}")
        split_df = split_df[split_df['site'].astype(str).isin(manifest_sites)].copy()
        if split_df.empty:
            raise ValueError("No split sites remain after patch-manifest site filtering.")
    selected_sites = sorted(split_df['site'].unique().tolist())
    split_role_by_site = dict(zip(split_df['site'], split_df['role']))

    with open(feature_sets_path, 'r') as fp:
        feature_sets = json.load(fp)
    requested_feature_set = data_cfg.get('feature_set_name', 'standard')
    requested_features = feature_sets[requested_feature_set]
    requested_features = list(dict.fromkeys(requested_features + data_cfg.get('extra_feature_columns', [])))

    target_schema = _read_schema_columns(target_path)
    modis_schema = _read_schema_columns(modis_path)
    era5_schema = _read_schema_columns(era5_path)

    target_time_column = _detect_time_column(target_schema)
    modis_time_column = _detect_time_column(modis_schema)
    era5_time_column = _detect_time_column(era5_schema)
    join_keys = ['site', 'date']

    target_columns = [
        'site', target_time_column, 'GPP_NT_VUT_USTAR50', 'RECO_NT_VUT_USTAR50',
        'NEE_VUT_USTAR50', 'NEE_VUT_USTAR50_QC', 'site', 'lat', 'lon', 'IGBP'
    ]
    target_columns = list(dict.fromkeys(target_columns))

    modis_feature_columns = [col for col in requested_features if col in modis_schema]
    era5_feature_columns = [col for col in requested_features if col in era5_schema]
    if not modis_feature_columns and not era5_feature_columns:
        raise ValueError(f"No requested CarbonBench features from set '{requested_feature_set}' were found in MODIS or ERA5 schemas.")

    modis_columns = list(dict.fromkeys(['site', modis_time_column] + modis_feature_columns))
    era5_columns = list(dict.fromkeys(['site', era5_time_column] + era5_feature_columns))

    target_df = _load_parquet_subset(target_path, target_columns, selected_sites)
    modis_df = _load_parquet_subset(modis_path, modis_columns, selected_sites)
    era5_df = _load_parquet_subset(era5_path, era5_columns, selected_sites)

    target_df = _normalize_timestamp_columns(target_df, target_time_column)
    modis_df = _normalize_timestamp_columns(modis_df, modis_time_column)
    era5_df = _normalize_timestamp_columns(era5_df, era5_time_column)

    flux_value_columns = ['GPP_NT_VUT_USTAR50', 'RECO_NT_VUT_USTAR50', 'NEE_VUT_USTAR50']
    for col in flux_value_columns:
        if col in target_df.columns:
            target_df[col] = target_df[col].where(target_df[col].between(-9000, 9000))

    target_df = target_df[target_df['date'].notna() & target_df['TIMESTAMP'].notna()].copy()
    modis_df = modis_df[modis_df['date'].notna()].copy()
    era5_df = era5_df[era5_df['date'].notna()].copy()

    merged = target_df.merge(modis_df, on=join_keys, how='inner', validate='one_to_one')
    merged = merged.merge(era5_df, on=join_keys, how='inner', validate='one_to_one')
    merged = _prepare_metadata(merged, koppen_sites_path)
    # Keep a full daily copy (all dates, no patch restriction) for dense sequence lookup.
    # This gives Mamba access to the real daily MODIS/ERA5 history rather than sparse patch-anchor dates.
    _use_seq = bool(data_cfg.get('use_sequence_branch', False))
    full_daily_merged: Optional[pd.DataFrame] = merged.copy() if _use_seq else None

    if patch_manifest_path:
        merged = _merge_patch_manifest(
            frame=merged,
            patch_manifest_path=patch_manifest_path,
            data_root=data_root,
            use_image_branch=bool(data_cfg.get('use_image_branch', False)),
            restrict_to_manifest_rows=restrict_to_manifest_rows,
            image_context_mode=image_context_mode,
            max_context_patches=image_context_max_patches,
        )
    igbp_filter = _resolve_igbp_filter(data_cfg.get('igbp_filter', 'forest'))
    if igbp_filter is not None:
        merged = merged[merged['IGBP'].isin(igbp_filter)].copy()

    target_column = data_cfg['target_column']
    target_columns = target_column if isinstance(target_column, list) else [target_column]
    merged = merged.dropna(subset=target_columns).copy()
    merged['role'] = merged['site'].map(split_role_by_site)
    merged = merged[merged['role'].isin(['source', 'target'])].copy()

    if full_daily_merged is not None:
        if igbp_filter is not None:
            full_daily_merged = full_daily_merged[full_daily_merged['IGBP'].isin(igbp_filter)].copy()
        full_daily_merged['role'] = full_daily_merged['site'].map(split_role_by_site)
        full_daily_merged = full_daily_merged[full_daily_merged['role'].isin(['source', 'target'])].copy()

    random_state = int(data_cfg.get('seed', 42))
    explicit_partition_map = None
    if 'split_partition' in split_df.columns:
        split_partition_df = split_df[['site', 'split_partition']].dropna().drop_duplicates()
        explicit_partition_map = dict(zip(split_partition_df['site'], split_partition_df['split_partition']))

    fixed_train_sites = data_cfg.get('fixed_train_sites')
    fixed_val_sites = data_cfg.get('fixed_val_sites')
    if explicit_partition_map is not None:
        merged = merged[merged['site'].isin(explicit_partition_map)].copy()
        merged['split_partition'] = merged['site'].map(explicit_partition_map)
        if full_daily_merged is not None:
            full_daily_merged = full_daily_merged[full_daily_merged['site'].isin(explicit_partition_map)].copy()
            full_daily_merged['split_partition'] = full_daily_merged['site'].map(explicit_partition_map)
        train_sites = {site for site, part in explicit_partition_map.items() if part == 'train'}
        val_sites = {site for site, part in explicit_partition_map.items() if part == 'val'}
        test_sites = {site for site, part in explicit_partition_map.items() if part == 'test'}
    elif fixed_train_sites is not None and fixed_val_sites is not None:
        train_sites = set(fixed_train_sites)
        val_sites = set(fixed_val_sites)
        test_sites = set(split_df.loc[split_df['role'] == 'target', 'site'].unique().tolist())
    else:
        source_sites = sorted(merged.loc[merged['role'] == 'source', 'site'].unique().tolist())
        val_fraction = float(data_cfg.get('source_val_fraction_sites', 0.2))
        if len(source_sites) < 2:
            raise ValueError("Need at least two source sites to create a train/validation split.")
        n_val_sites = max(1, int(round(len(source_sites) * val_fraction)))
        if n_val_sites >= len(source_sites):
            n_val_sites = len(source_sites) - 1
        train_sites, val_sites = train_test_split(source_sites, test_size=n_val_sites, random_state=random_state)
        train_sites = set(train_sites)
        val_sites = set(val_sites)
        test_sites = set(merged.loc[merged['role'] == 'target', 'site'].unique().tolist())

    if explicit_partition_map is None:
        merged['split_partition'] = np.where(
            merged['site'].isin(train_sites), 'train',
            np.where(merged['site'].isin(val_sites), 'val', 'test')
        )
        if full_daily_merged is not None:
            full_daily_merged['split_partition'] = np.where(
                full_daily_merged['site'].isin(train_sites), 'train',
                np.where(full_daily_merged['site'].isin(val_sites), 'val', 'test')
            )

    train_df = merged[merged['split_partition'] == 'train'].copy()
    val_df = merged[merged['split_partition'] == 'val'].copy()
    test_df = merged[merged['split_partition'] == 'test'].copy()

    eval_qc_threshold = data_cfg.get('eval_qc_threshold')
    if eval_qc_threshold is not None and 'NEE_VUT_USTAR50_QC' in merged.columns:
        threshold = float(eval_qc_threshold)
        val_df = val_df[val_df['NEE_VUT_USTAR50_QC'].fillna(0.0) >= threshold].copy()
        test_df = test_df[test_df['NEE_VUT_USTAR50_QC'].fillna(0.0) >= threshold].copy()

    sequence_stride = int(data_cfg.get('sequence_stride', 1) or 1)
    if sequence_stride > 1:
        train_df = _apply_temporal_stride(train_df, sequence_stride)
        val_df = _apply_temporal_stride(val_df, sequence_stride)
        test_df = _apply_temporal_stride(test_df, sequence_stride)

    max_train_rows = data_cfg.get('max_train_rows')
    max_val_rows = data_cfg.get('max_val_rows')
    max_test_rows = data_cfg.get('max_test_rows')
    if max_train_rows:
        train_df = train_df.sample(n=min(max_train_rows, len(train_df)), random_state=random_state)
    if max_val_rows:
        val_df = val_df.sample(n=min(max_val_rows, len(val_df)), random_state=random_state)
    if max_test_rows:
        test_df = test_df.sample(n=min(max_test_rows, len(test_df)), random_state=random_state)

    categorical_cols = ['IGBP', 'koppen_main']
    categories_by_col = {
        'IGBP': sorted(merged['IGBP'].fillna('UNK').unique().tolist()),
        'koppen_main': sorted(merged['koppen_main'].fillna('UNK').unique().tolist()),
    }
    numeric_metadata_cols = ['lat', 'lon', 'doy_sin', 'doy_cos']

    for frame_name, frame in [('train', train_df), ('val', val_df), ('test', test_df)]:
        encoded = _one_hot_encode(frame, categorical_cols, categories_by_col)
        frame.loc[:, numeric_metadata_cols] = frame[numeric_metadata_cols].astype(np.float32)
        frame_metadata = pd.concat([frame[numeric_metadata_cols], encoded], axis=1)
        frame_metadata.columns = frame_metadata.columns.astype(str)
        for col in frame_metadata.columns:
            frame[col] = frame_metadata[col]
        if frame_name == 'train':
            train_df = frame
        elif frame_name == 'val':
            val_df = frame
        else:
            test_df = frame

    metadata_columns = numeric_metadata_cols + [
        col for col in train_df.columns if col.startswith('IGBP__') or col.startswith('koppen_main__')
    ]
    tabular_columns = modis_feature_columns + era5_feature_columns
    sequence_extra_feature_columns = [
        col for col in data_cfg.get('sequence_extra_feature_columns', [])
        if col in train_df.columns
    ]
    sequence_columns = tabular_columns + [col for col in sequence_extra_feature_columns if col not in tabular_columns]

    train_df, [val_df, test_df], _train_mean, _train_std, _train_fill = _standardize_split(
        train_df, [val_df, test_df], tabular_columns + metadata_columns
    )

    sample_weight_column = None
    if bool(data_cfg.get('use_carbonbench_sample_weights', False)):
        train_df, [val_df, test_df] = _add_sample_weights(train_df, [val_df, test_df])
        sample_weight_column = 'sample_weight'

    if bool(data_cfg.get('standardize_targets', False)):
        target_mean = train_df[target_columns].mean()
        target_std = train_df[target_columns].std().replace(0, 1.0).fillna(1.0)
        for frame in [train_df, val_df, test_df]:
            frame.loc[:, target_columns] = (frame[target_columns] - target_mean) / target_std
        config['data']['target_mean'] = {col: float(target_mean[col]) for col in target_columns}
        config['data']['target_std'] = {col: float(target_std[col]) for col in target_columns}
        config['data']['target_columns_resolved'] = target_columns
    if full_daily_merged is not None:
        _seq_norm_cols = [c for c in tabular_columns + ['doy_sin', 'doy_cos'] if c in full_daily_merged.columns]
        _seq_fill = _train_fill.reindex(_seq_norm_cols)
        _seq_mean = _train_mean.reindex(_seq_norm_cols)
        _seq_std = _train_std.reindex(_seq_norm_cols).replace(0, 1.0).fillna(1.0)
        full_daily_merged = full_daily_merged.copy()
        full_daily_merged[_seq_norm_cols] = (
            full_daily_merged[_seq_norm_cols]
            .fillna(_seq_fill)
            .sub(_seq_mean)
            .div(_seq_std)
        )

    config['model']['params']['tabular_input_dim'] = len(tabular_columns)
    config['model']['params']['metadata_input_dim'] = len(metadata_columns)
    config['model']['params']['sequence_input_dim'] = len(sequence_columns)
    config['model']['params']['use_sequence_branch'] = bool(data_cfg.get('use_sequence_branch', False))
    config['model']['params']['use_image_branch'] = bool(data_cfg.get('use_image_branch', False))
    config['model']['params']['image_channels'] = int(data_cfg.get('image_channels', 0))
    config['data']['tabular_columns'] = tabular_columns
    config['data']['metadata_columns'] = metadata_columns
    config['data']['sequence_columns'] = sequence_columns
    config['data']['resolved_split_file'] = str(split_path)
    config['data']['resolved_data_root'] = str(data_root)
    config['data']['train_sites'] = sorted(train_sites)
    config['data']['val_sites'] = sorted(val_sites)
    config['data']['test_sites'] = sorted(test_sites)

    include_patch_fields = bool(data_cfg.get('include_patch_fields', True))
    sequence_length = int(data_cfg.get('sequence_length', 0))
    sequence_include_current = bool(data_cfg.get('sequence_include_current', True))
    dataset_sequence_columns = sequence_columns if bool(data_cfg.get('use_sequence_branch', False)) else []
    train_dataset = CarbonBenchFluxDataset(
        train_df,
        tabular_columns,
        metadata_columns,
        target_column,
        include_patch_fields,
        sequence_columns=dataset_sequence_columns,
        sequence_length=sequence_length,
        sequence_include_current=sequence_include_current,
        sequence_frame=full_daily_merged,
        sample_weight_column=sample_weight_column,
        max_context_patches=image_context_max_patches,
        patch_shape=(int(data_cfg.get('image_channels', 6)), 67, 67),
    )
    val_dataset = CarbonBenchFluxDataset(
        val_df,
        tabular_columns,
        metadata_columns,
        target_column,
        include_patch_fields,
        sequence_columns=dataset_sequence_columns,
        sequence_length=sequence_length,
        sequence_include_current=sequence_include_current,
        sequence_frame=full_daily_merged,
        sample_weight_column=sample_weight_column,
        max_context_patches=image_context_max_patches,
        patch_shape=(int(data_cfg.get('image_channels', 6)), 67, 67),
    )
    test_dataset = CarbonBenchFluxDataset(
        test_df,
        tabular_columns,
        metadata_columns,
        target_column,
        include_patch_fields,
        sequence_columns=dataset_sequence_columns,
        sequence_length=sequence_length,
        sequence_include_current=sequence_include_current,
        sequence_frame=full_daily_merged,
        sample_weight_column=sample_weight_column,
        max_context_patches=image_context_max_patches,
        patch_shape=(int(data_cfg.get('image_channels', 6)), 67, 67),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=True,
        num_workers=data_cfg['num_workers'],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.get('val_batch_size', data_cfg['batch_size']),
        shuffle=False,
        num_workers=data_cfg['num_workers'],
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg.get('test_batch_size', data_cfg['batch_size']),
        shuffle=False,
        num_workers=data_cfg['num_workers'],
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"CarbonBench flux dataloaders ready. "
        f"Train rows={len(train_dataset)}, Val rows={len(val_dataset)}, Test rows={len(test_dataset)} | "
        f"Tabular dims={len(tabular_columns)}, Metadata dims={len(metadata_columns)}, "
        f"Sequence dims={len(dataset_sequence_columns)} x {sequence_length if dataset_sequence_columns else 0} | "
        f"Sequence source={'full_daily_history' if dataset_sequence_columns and full_daily_merged is not None else 'sample_rows'} | "
        f"Image context={image_context_mode} | K={image_context_max_patches}"
    )

    return train_loader, val_loader, test_loader
