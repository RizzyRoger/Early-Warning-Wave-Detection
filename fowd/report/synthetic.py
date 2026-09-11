"""
synthetic.py

Synthetic FOWD-like catalogue with crest-trough correlation as the planted driver.
"""

import numpy as np
import pandas as pd

from .features import add_labels, hs_column, normalize_interval, sea_state_prefix


def generate_catalogue(n_waves=8000, seed=0, interval='30m'):
    """Return a DataFrame that looks like a loaded FOWD catalogue."""
    interval = normalize_interval(interval)
    prefix = sea_state_prefix(interval)
    rng = np.random.RandomState(seed)

    n1 = n_waves // 2
    n2 = n_waves - n1
    station = np.array(['SYNTH_A'] * n1 + ['SYNTH_B'] * n2)

    start = np.datetime64('2018-01-01T00:00:00')
    offsets_a = np.arange(n1) * np.timedelta64(12, 's')
    offsets_b = np.arange(n2) * np.timedelta64(12, 's')
    wave_start = np.concatenate([start + offsets_a, start + offsets_b])
    wave_end = wave_start + np.timedelta64(10, 's')

    r = rng.uniform(0.05, 0.95, n_waves)
    steepness = rng.normal(0.04, 0.01, n_waves).clip(0.01, 0.12)
    bandwidth = rng.normal(0.4, 0.1, n_waves).clip(0.1, 0.9)
    hs = rng.uniform(1.2, 4.5, n_waves)
    mean_period = rng.uniform(6.0, 14.0, n_waves)
    peak_period = mean_period * rng.uniform(1.05, 1.35, n_waves)
    depth = np.where(station == 'SYNTH_A', 80.0, 400.0)

    logit = -4.4 + 4.6 * r + 4.0 * (steepness - 0.04) + rng.normal(0.0, 0.25, n_waves)
    prob = 1.0 / (1.0 + np.exp(-logit))
    is_rogue = rng.rand(n_waves) < prob

    rel_height = np.where(
        is_rogue,
        rng.uniform(2.05, 2.8, n_waves),
        rng.uniform(0.6, 1.85, n_waves),
    )
    wave_height = rel_height * hs

    df = pd.DataFrame({
        'meta_station_name': station,
        'wave_start_time': wave_start,
        'wave_end_time': wave_end,
        'meta_deploy_latitude': np.where(station == 'SYNTH_A', 34.5, 46.2),
        'meta_deploy_longitude': np.where(station == 'SYNTH_A', -120.7, -124.1),
        'meta_water_depth': depth,
        'wave_height': wave_height,
        '{}_significant_wave_height_spectral'.format(prefix): hs,
        '{}_significant_wave_height_direct'.format(prefix): hs * rng.uniform(0.92, 1.08, n_waves),
        '{}_mean_period_spectral'.format(prefix): mean_period,
        '{}_mean_period_direct'.format(prefix): mean_period * rng.uniform(0.95, 1.05, n_waves),
        '{}_peak_wave_period'.format(prefix): peak_period,
        '{}_steepness'.format(prefix): steepness,
        '{}_skewness'.format(prefix): rng.normal(0.1, 0.15, n_waves),
        '{}_kurtosis'.format(prefix): rng.normal(0.2, 0.4, n_waves),
        '{}_bandwidth_peakedness'.format(prefix): bandwidth,
        '{}_bandwidth_narrowness'.format(prefix): bandwidth * rng.uniform(0.85, 1.15, n_waves),
        '{}_benjamin_feir_index_peakedness'.format(prefix): steepness / np.maximum(bandwidth, 0.15),
        '{}_benjamin_feir_index_narrowness'.format(prefix): steepness / np.maximum(bandwidth, 0.15),
        '{}_crest_trough_correlation'.format(prefix): r,
        '{}_rel_maximum_wave_height'.format(prefix): rng.uniform(1.3, 2.2, n_waves),
        '{}_valid_data_ratio'.format(prefix): rng.uniform(0.96, 1.0, n_waves),
    })

    labeled = add_labels(df, interval=interval)
    if int(labeled['is_rogue'].sum()) < 20:
        top_r = labeled['{}_crest_trough_correlation'.format(prefix)].nlargest(20).index
        hs_name = hs_column(interval)
        labeled.loc[top_r, 'relative_wave_height'] = 2.2
        labeled.loc[top_r, 'is_rogue'] = 1
        labeled.loc[top_r, 'wave_height'] = 2.2 * labeled.loc[top_r, hs_name]
    return labeled
