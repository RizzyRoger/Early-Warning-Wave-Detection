"""
tests/test_worldmap.py

Cartopy world map PNG + Pacific placement.
"""

import os

import pandas as pd
import pytest

from fowd.report import store
from fowd.report.maprender import resolve_frame, time_bins
from fowd.report.worldmap import (
    MapViewer,
    collected_projected,
    draw_world_map,
    forecast_split_index,
    interactive_world_map,
    require_cartopy,
)


def _places():
    return pd.DataFrame({
        'meta_station_name': ['SYNTH_A', 'SYNTH_B'],
        'meta_deploy_latitude': [34.5, 46.2],
        'meta_deploy_longitude': [-120.7, -124.1],
        'wave_start_time': pd.to_datetime([
            '2018-01-01 00:10:00',
            '2018-01-01 01:20:00',
        ]),
        'is_rogue': [1, 1],
        'rogue_prob': [0.9, 0.4],
        'split': ['train', 'test'],
    })


def test_draw_world_map_writes_png_pacific(tmp_path):
    try:
        require_cartopy()
    except RuntimeError:
        pytest.skip('cartopy is not installed')

    places = _places()
    assert places['meta_deploy_longitude'].max() < -100
    path = os.path.join(str(tmp_path), 'world_map.png')
    out = draw_world_map(places, 'test rogue map', path, show=False)
    assert out == path
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 1000


def test_resolve_frame_next_updates_state(tmp_path):
    out_dir = str(tmp_path)
    run_id, run_path = store.create_run(out_dir, source='test')
    _places().to_csv(os.path.join(run_path, 'predictions.csv'), index=False)
    store.update_run(out_dir, run_id, status='ok')

    first = resolve_frame(out_dir, run_id=run_id, bin_size='1h', step=1)
    assert first['state']['bin_index'] == 0
    second = resolve_frame(out_dir, run_id=run_id, bin_size='1h', step=1)
    assert second['state']['bin_index'] == 1
    assert os.path.isfile(os.path.join(run_path, 'map_state.json'))


def test_train_test_slider_split_and_disjoint_markers():
    places = _places()
    bins = time_bins(places, bin_size='1h')
    assert len(bins) == 2
    assert forecast_split_index(places, bins) == 1

    collected, projected = collected_projected(places)
    assert len(collected) == 1
    assert len(projected) == 1
    assert set(collected.index).isdisjoint(set(projected.index))
    assert (collected['split'] == 'train').all()
    assert (projected['split'] == 'test').all()

    hour0_c, hour0_p = collected_projected(bins[0][2])
    hour1_c, hour1_p = collected_projected(bins[1][2])
    assert len(hour0_c) == 1 and hour0_p.empty
    assert hour1_c.empty and len(hour1_p) == 1


def test_map_viewer_slider_and_zoom(tmp_path):
    try:
        require_cartopy()
    except RuntimeError:
        pytest.skip('cartopy is not installed')

    places = _places()
    path = os.path.join(str(tmp_path), 'world_map.png')
    viewer = MapViewer(places, 'run-test', path, bin_size='1h', show=False)
    assert viewer.split_index == 1
    assert len(viewer.bins) == 2

    viewer.set_index(0, save=False)
    assert viewer.sc_collected.get_offsets().shape[0] == 1
    assert viewer.sc_projected.get_offsets().shape[0] == 0

    viewer.set_index(1, save=False)
    assert viewer.sc_collected.get_offsets().shape[0] == 0
    assert viewer.sc_projected.get_offsets().shape[0] == 1

    x0, x1, y0, y1 = viewer.ax.get_extent(crs=viewer.proj)
    viewer.zoom_at(-120.0, 40.0, 0.5)
    nx0, nx1, ny0, ny1 = viewer.ax.get_extent(crs=viewer.proj)
    assert (nx1 - nx0) < (x1 - x0)
    assert (ny1 - ny0) < (y1 - y0)

    viewer.set_index(0, save=True)
    assert os.path.isfile(path)

    import matplotlib.pyplot as plt
    plt.close(viewer.fig)


def test_interactive_no_show_last_collected_bin(tmp_path):
    try:
        require_cartopy()
    except RuntimeError:
        pytest.skip('cartopy is not installed')

    path = os.path.join(str(tmp_path), 'world_map.png')
    viewer = interactive_world_map(
        _places(), 'run-test', path, show=False, bin_size='1h',
    )
    assert viewer.index == 0
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 1000


def test_cli_map_no_show_and_next(tmp_path):
    try:
        require_cartopy()
    except RuntimeError:
        pytest.skip('cartopy is not installed')

    from click.testing import CliRunner
    from fowd.cli import cli

    out_dir = str(tmp_path)
    run_id, run_path = store.create_run(out_dir, source='test')
    _places().to_csv(os.path.join(run_path, 'predictions.csv'), index=False)
    store.update_run(out_dir, run_id, status='ok')

    runner = CliRunner()
    result = runner.invoke(cli, [
        'report', 'map', '--no-show', '-o', out_dir, '--run', run_id,
    ])
    assert result.exit_code == 0, result.output
    png = os.path.join(run_path, 'world_map.png')
    assert os.path.isfile(png)

    stepped = runner.invoke(cli, [
        'report', 'map', '--no-show', '-o', out_dir, '--run', run_id, 'next',
    ])
    assert stepped.exit_code == 0, stepped.output
    assert os.path.isfile(png)
