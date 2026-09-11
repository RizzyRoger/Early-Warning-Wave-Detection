"""
filter.py

Rank scored wave windows by predicted rogue-wave probability.
"""

import os

import pandas as pd

from . import store


PREVIEW_COLUMNS = (
    'meta_station_name',
    'wave_start_time',
    'meta_deploy_latitude',
    'meta_deploy_longitude',
    'meta_water_depth',
    'wave_height',
    'relative_wave_height',
    'rogue_prob',
    'is_rogue',
    'split',
)


def load_predictions(out_dir, run_id=None):
    run_id = store.resolve_run_id(out_dir, run_id)
    path = os.path.join(store.run_dir(out_dir, run_id), 'predictions.csv')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            'No predictions.csv for run {}. Generate a report first.'.format(run_id)
        )
    return run_id, pd.read_csv(path)


def filter_rogue_places(predictions, top=20, min_prob=0.0, holdout_only=False,
                        actual_only=False):
    """Return high-risk or confirmed-rogue places and a small summary dict."""
    table = predictions.copy()
    if holdout_only and 'split' in table.columns:
        table = table.loc[table['split'] == 'test']
    if actual_only:
        if 'is_rogue' not in table.columns:
            raise ValueError('predictions are missing is_rogue')
        table = table.loc[table['is_rogue'] == 1]
    if 'rogue_prob' not in table.columns:
        raise ValueError('predictions are missing rogue_prob')

    table = table.loc[table['rogue_prob'] >= float(min_prob)]
    if actual_only and 'relative_wave_height' in table.columns:
        table = table.sort_values(
            ['relative_wave_height', 'rogue_prob'], ascending=False
        ).head(int(top))
    else:
        table = table.sort_values('rogue_prob', ascending=False).head(int(top))

    n = len(table)
    n_true = int(table['is_rogue'].sum()) if 'is_rogue' in table.columns and n else 0
    summary = {
        'n_returned': n,
        'min_prob': float(min_prob),
        'holdout_only': bool(holdout_only),
        'actual_only': bool(actual_only),
        'precision_at_n': (float(n_true) / float(n)) if n else 0.0,
        'n_actual_rogues': n_true,
        'max_rogue_prob': float(table['rogue_prob'].max()) if n else 0.0,
    }
    return table.reset_index(drop=True), summary


def format_places(table, summary, run_id):
    lines = [
        'High-risk places for run {}'.format(run_id),
        'Showing {} rows with P(rogue) >= {:.3f}{}{}'.format(
            summary['n_returned'],
            summary['min_prob'],
            ' (hold-out only)' if summary['holdout_only'] else '',
            ' (confirmed rogues only)' if summary.get('actual_only') else '',
        ),
        'Precision@N: {:.3f} ({}/{} actually rogue)'.format(
            summary['precision_at_n'],
            summary['n_actual_rogues'],
            summary['n_returned'],
        ),
        '',
    ]
    if table.empty:
        lines.append('No rows matched the filter.')
        return '\n'.join(lines)

    show = [col for col in PREVIEW_COLUMNS if col in table.columns]
    view = table[show]
    with pd.option_context('display.max_columns', None, 'display.width', 160):
        lines.append(view.to_string(index=False))
    return '\n'.join(lines)
