"""
tests/test_report_pipeline.py

Synthetic generate + rogue-place filter.
"""

import os

import pandas as pd
import pytest

from fowd.report.filter import filter_rogue_places
from fowd.report.pipeline import PipelineError, generate_report


def test_filter_returns_highest_rogue_prob():
    predictions = pd.DataFrame({
        'meta_station_name': ['A', 'B', 'C', 'D'],
        'rogue_prob': [0.1, 0.9, 0.4, 0.8],
        'is_rogue': [0, 1, 0, 1],
        'split': ['test', 'test', 'train', 'test'],
    })
    table, summary = filter_rogue_places(predictions, top=2, min_prob=0.5)
    assert list(table['rogue_prob']) == [0.9, 0.8]
    assert summary['n_returned'] == 2
    assert summary['n_actual_rogues'] == 2
    assert summary['precision_at_n'] == 1.0

    holdout, holdout_summary = filter_rogue_places(
        predictions, top=10, min_prob=0.0, holdout_only=True
    )
    assert set(holdout['meta_station_name']) == {'A', 'B', 'D'}
    assert holdout_summary['holdout_only'] is True

    actual, actual_summary = filter_rogue_places(predictions, top=10, actual_only=True)
    assert set(actual['is_rogue']) == {1}
    assert actual_summary['actual_only'] is True


def test_generate_synthetic_and_filter(tmp_path):
    pytest.importorskip('sklearn')

    out_dir = str(tmp_path)
    result = generate_report(out_dir, synthetic=True, seed=0)
    report = result['report']

    assert report['status'] == 'ok'
    assert 'pr_auc' in report
    assert report['pr_auc'] >= report['baseline_pr_auc'] - 1e-9

    run_dir = result['run_dir']
    assert os.path.isfile(os.path.join(run_dir, 'importances.csv'))
    assert os.path.isfile(os.path.join(run_dir, 'predictions.csv'))
    assert os.path.isfile(os.path.join(run_dir, 'report.md'))
    assert os.path.isfile(os.path.join(run_dir, 'report.json'))

    predictions = pd.read_csv(os.path.join(run_dir, 'predictions.csv'))
    assert 'rogue_prob' in predictions.columns
    table, summary = filter_rogue_places(predictions, top=10)
    assert summary['n_returned'] == 10
    assert table['rogue_prob'].is_monotonic_decreasing
    assert table['rogue_prob'].iloc[0] == predictions['rogue_prob'].max()


def test_long_train_writes_checkpoints(tmp_path):
    pytest.importorskip('sklearn')
    out_dir = str(tmp_path)
    result = generate_report(
        out_dir, synthetic=True, seed=0, rounds=12, save_every=5, n_waves=400
    )
    ckpt = os.path.join(result['run_dir'], 'checkpoints')
    saved = sorted(os.listdir(ckpt)) if os.path.isdir(ckpt) else []
    assert any(name.startswith('round_000005') for name in saved)
    assert any(name.startswith('round_000010') for name in saved)


def test_generate_rejects_multiple_sources(tmp_path):
    with pytest.raises(PipelineError):
        generate_report(
            str(tmp_path),
            input_files=['missing.nc'],
            synthetic=True,
        )


def test_resolve_run_id_skips_failed_without_predictions(tmp_path):
    from fowd.report import store

    out_dir = str(tmp_path)
    ok_id, ok_dir = store.create_run(out_dir, source='ok')
    pd.DataFrame({'rogue_prob': [0.1]}).to_csv(
        os.path.join(ok_dir, 'predictions.csv'), index=False
    )
    store.update_run(out_dir, ok_id, status='ok')
    failed_id, _ = store.create_run(out_dir, source='cdip')
    store.update_run(out_dir, failed_id, status='failed')
    assert store.resolve_run_id(out_dir) == ok_id
