"""
features.py

Sea-state feature lists, QC cuts, and rogue-wave labels.
"""

FILL_VALUE = -9999
HS_MIN_METERS = 1.0
VALID_DATA_RATIO_MIN = 0.95
ROGUE_THRESHOLD = 2.0

PLACE_COLUMNS = (
    'meta_station_name',
    'wave_start_time',
    'wave_end_time',
    'meta_deploy_latitude',
    'meta_deploy_longitude',
    'wave_height',
)

MODEL_FEATURE_SUFFIXES = (
    'significant_wave_height_spectral',
    'significant_wave_height_direct',
    'mean_period_spectral',
    'mean_period_direct',
    'peak_wave_period',
    'steepness',
    'skewness',
    'kurtosis',
    'bandwidth_peakedness',
    'bandwidth_narrowness',
    'benjamin_feir_index_peakedness',
    'benjamin_feir_index_narrowness',
    'crest_trough_correlation',
    'rel_maximum_wave_height',
)


def normalize_interval(interval):
    """Normalize a sea-state interval token (10, '10m', 'dynamic')."""
    if interval is None:
        return '30m'
    if not isinstance(interval, str):
        return '{}m'.format(interval)
    interval = interval.strip()
    if interval.isdigit():
        return '{}m'.format(interval)
    return interval


def sea_state_prefix(interval='30m'):
    return 'sea_state_{}'.format(normalize_interval(interval))


def hs_column(interval='30m'):
    return '{}_significant_wave_height_spectral'.format(sea_state_prefix(interval))


def valid_ratio_column(interval='30m'):
    return '{}_valid_data_ratio'.format(sea_state_prefix(interval))


def feature_columns(interval='30m'):
    """Model features for a sea-state aggregation window."""
    prefix = sea_state_prefix(interval)
    cols = ['{}_{}'.format(prefix, suffix) for suffix in MODEL_FEATURE_SUFFIXES]
    cols.append('meta_water_depth')
    return cols


def load_columns(interval='30m'):
    """Columns to pull from a FOWD netCDF (features + place metadata + QC)."""
    cols = list(feature_columns(interval))
    cols.append(valid_ratio_column(interval))
    for name in PLACE_COLUMNS:
        if name not in cols:
            cols.append(name)
    return cols


def apply_qc(df, interval='30m'):
    """Drop invalid rows. Returns (cleaned_frame, dropped_count)."""
    import numpy as np

    work = df.copy()
    work = work.replace(FILL_VALUE, np.nan)

    hs = hs_column(interval)
    ratio = valid_ratio_column(interval)
    before = len(work)

    mask = work['wave_height'].notna() & work[hs].notna() & (work[hs] >= HS_MIN_METERS)
    if ratio in work.columns:
        mask = mask & (work[ratio].isna() | (work[ratio] >= VALID_DATA_RATIO_MIN))

    work = work.loc[mask].reset_index(drop=True)
    return work, before - len(work)


def add_labels(df, interval='30m'):
    """Add relative height and rogue label columns."""
    hs = hs_column(interval)
    out = df.copy()
    out['relative_wave_height'] = out['wave_height'] / out[hs]
    out['is_rogue'] = (out['relative_wave_height'] >= ROGUE_THRESHOLD).astype(int)
    return out
