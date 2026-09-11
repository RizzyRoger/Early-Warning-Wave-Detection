"""
tests/test_report_map.py

World map placement, probability glyphs, and next/prev state.
"""

import json
import os

import pandas as pd

from fowd.report import store
from fowd.report.maprender import (
    WORLD_BOUNDS,
    _project,
    bin_index_for_time,
    build_view,
    flagged_places,
    glyph_for_prob,
    render_ascii,
    time_bins,
)


def _places():
    return pd.DataFrame({
        'meta_station_name': ['SYNTH_A', 'SYNTH_B', 'SYNTH_A'],
        'meta_deploy_latitude': [34.5, 46.2, 34.5],
        'meta_deploy_longitude': [-120.7, -124.1, -120.7],
        'wave_start_time': pd.to_datetime([
            '2018-01-01 00:10:00',
            '2018-01-01 01:20:00',
            '2018-01-01 01:40:00',
        ]),
        'is_rogue': [1, 1, 0],
        'rogue_prob': [0.9, 0.8, 0.7],
        'split': ['train', 'train', 'test'],
    })


def test_time_bins_split_two_hours():
    bins = time_bins(_places(), bin_size='1h')
    assert len(bins) >= 2
    first = bins[0][2]
    second = bins[1][2]
    assert len(first) == 1
    assert len(second) == 2
    assert bin_index_for_time(bins, '2018-01-01 01:15:00') == 1


def test_glyphs_grow_with_probability():
    assert glyph_for_prob(0.1) == '.'
    assert glyph_for_prob(0.3) == 'o'
    assert glyph_for_prob(0.6) == 'O'
    assert glyph_for_prob(0.9) == '@'


def test_render_world_puts_pacific_west_of_africa():
    places = _places()
    width, height = 40, 12
    canvas, bounds = render_ascii(places, bounds=WORLD_BOUNDS, width=width, height=height)
    assert bounds == WORLD_BOUNDS
    assert '@' in canvas

    body = canvas.splitlines()
    assert len(body) == height
    assert all(len(line) == width for line in body)

    pacific_cols = []
    for lat, lon in ((34.5, -120.7), (46.2, -124.1)):
        row, col = _project(lon, lat, WORLD_BOUNDS, width, height)
        pacific_cols.append(col)
        assert body[row][col] in '@O'
    africa_col = _project(20.0, 10.0, WORLD_BOUNDS, width, height)[1]
    asia_col = _project(100.0, 30.0, WORLD_BOUNDS, width, height)[1]
    assert max(pacific_cols) < africa_col
    assert max(pacific_cols) < asia_col


def test_map_next_advances_state(tmp_path):
    out_dir = str(tmp_path)
    run_id, run_path = store.create_run(out_dir, source='test')
    _places().to_csv(os.path.join(run_path, 'predictions.csv'), index=False)
    store.update_run(out_dir, run_id, status='ok')

    _, text0, state0 = build_view(
        out_dir, run_id=run_id, bin_size='1h', step=1, width=40, height=12
    )
    assert 'Frame: 1 /' in text0 or 'frame 1/' in text0
    assert state0['bin_index'] == 0

    _, text1, state1 = build_view(
        out_dir, run_id=run_id, bin_size='1h', step=1, width=40, height=12
    )
    assert state1['bin_index'] == 1
    assert 'frame 2/' in text1

    path = os.path.join(run_path, 'map_state.json')
    with open(path, 'r') as handle:
        saved = json.load(handle)
    assert saved['bin_index'] == 1

    _, text_prev, state_prev = build_view(
        out_dir, run_id=run_id, bin_size='1h', step=-1, width=40, height=12
    )
    assert state_prev['bin_index'] == 0
    assert 'frame 1/' in text_prev
    flagged = flagged_places(_places(), actual_only=True)
    assert set(flagged['is_rogue']) == {1}
