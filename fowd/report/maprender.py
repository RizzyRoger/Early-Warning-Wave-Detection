"""
maprender.py

Full-terminal world map of flagged rogue-wave places.
"""

import json
import os
import shutil

import pandas as pd

from . import store
from .filter import load_predictions

STATE_NAME = 'map_state.json'

WORLD_BOUNDS = {
    'lon0': -180.0,
    'lon1': 180.0,
    'lat0': -90.0,
    'lat1': 90.0,
}

# Crude continent boxes (lon0, lon1, lat0, lat1) for a recognizable world silhouette.
LAND_BOXES = (
    (-168, -52, 15, 72),      # North America
    (-117, -86, 14, 32),      # Mexico / Central America
    (-82, -34, -56, 12),      # South America
    (-73, -12, 60, 84),       # Greenland
    (-10, 40, 36, 71),        # Europe
    (-18, 51, -35, 37),       # Africa
    (26, 145, 8, 77),         # Asia
    (145, 180, 42, 75),       # far-east Russia
    (95, 141, -11, 6),        # Indonesia / New Guinea
    (113, 154, -44, -10),     # Australia
    (165, 179, -47, -34),     # New Zealand
    (-180, 180, -90, -65),    # Antarctica
)

HEADER_RESERVE = 2
FOOTER_RESERVE = 2

BIN_ALIASES = {
    '30min': pd.Timedelta(minutes=30),
    '30m': pd.Timedelta(minutes=30),
    '1h': pd.Timedelta(hours=1),
    '6h': pd.Timedelta(hours=6),
}


def parse_bin(bin_size):
    key = str(bin_size).strip().lower()
    if key not in BIN_ALIASES:
        raise ValueError('Unknown --bin {!r}. Use 30min, 1h, or 6h.'.format(bin_size))
    return BIN_ALIASES[key], key


def parse_time(value):
    stamp = pd.to_datetime(value, utc=False)
    if pd.isna(stamp):
        raise ValueError('Could not parse time {!r}'.format(value))
    return stamp


def _as_time_series(series):
    return pd.to_datetime(series, utc=False)


def flagged_places(predictions, min_prob=0.0, holdout_only=False, actual_only=False):
    """All flagged rows (no top-N cut) for mapping."""
    table = predictions.copy()
    if holdout_only and 'split' in table.columns:
        table = table.loc[table['split'] == 'test']
    if actual_only:
        if 'is_rogue' not in table.columns:
            raise ValueError('predictions are missing is_rogue')
        table = table.loc[table['is_rogue'] == 1]
    if 'rogue_prob' in table.columns:
        table = table.loc[table['rogue_prob'] >= float(min_prob)]
    elif not actual_only:
        raise ValueError('predictions are missing rogue_prob')

    needed = ['meta_deploy_latitude', 'meta_deploy_longitude', 'wave_start_time']
    missing = [name for name in needed if name not in table.columns]
    if missing:
        raise ValueError('predictions missing {}'.format(', '.join(missing)))

    table = table.dropna(subset=needed)
    if 'wave_start_time' in table.columns:
        table = table.copy()
        table['wave_start_time'] = _as_time_series(table['wave_start_time'])
    return table.reset_index(drop=True)


def is_land(lon, lat):
    """True if the point sits in a baked-in continent box."""
    for lon0, lon1, lat0, lat1 in LAND_BOXES:
        if lon0 <= lon <= lon1 and lat0 <= lat <= lat1:
            return True
    return False


def terminal_canvas_size(width=0, height=0):
    """Map body size. 0 means fill the current terminal."""
    cols, rows = shutil.get_terminal_size(fallback=(80, 24))
    if not width:
        width = cols
    if not height:
        height = rows - HEADER_RESERVE - FOOTER_RESERVE
    return max(40, int(width)), max(10, int(height))


def glyph_for_prob(prob):
    """Bigger character = higher P(rogue)."""
    value = float(prob)
    if value >= 0.75:
        return '@'
    if value >= 0.50:
        return 'O'
    if value >= 0.25:
        return 'o'
    return '.'


def aggregate_places(places):
    """One row per lat/lon: max rogue_prob and whether any event was a rogue."""
    if places.empty:
        return places
    work = places.copy()
    keys = ['meta_deploy_latitude', 'meta_deploy_longitude']
    agg = {}
    if 'rogue_prob' in work.columns:
        agg['rogue_prob'] = 'max'
    if 'is_rogue' in work.columns:
        agg['is_rogue'] = 'max'
    if 'meta_station_name' in work.columns:
        agg['meta_station_name'] = 'first'
    if not agg:
        return work[keys].drop_duplicates().reset_index(drop=True)
    return work.groupby(keys, as_index=False).agg(agg)


def geo_bounds(places, pad=0.15):
    """Lon/lat box covering places, padded so two stations do not sit on the rim."""
    lons = places['meta_deploy_longitude'].astype(float)
    lats = places['meta_deploy_latitude'].astype(float)
    lon0, lon1 = float(lons.min()), float(lons.max())
    lat0, lat1 = float(lats.min()), float(lats.max())
    if lon0 == lon1:
        lon0 -= 1.0
        lon1 += 1.0
    if lat0 == lat1:
        lat0 -= 1.0
        lat1 += 1.0
    dlon = lon1 - lon0
    dlat = lat1 - lat0
    return {
        'lon0': lon0 - pad * dlon,
        'lon1': lon1 + pad * dlon,
        'lat0': lat0 - pad * dlat,
        'lat1': lat1 + pad * dlat,
    }


def time_bins(places, bin_size='1h'):
    """Return [(start, end, frame), ...] covering flagged events."""
    delta, _ = parse_bin(bin_size)
    if places.empty:
        return []
    times = _as_time_series(places['wave_start_time'])
    start = times.min().floor(delta)
    end = times.max().ceil(delta)
    if end <= start:
        end = start + delta
    edges = pd.date_range(start, end, freq=delta)
    if len(edges) < 2:
        edges = pd.DatetimeIndex([start, start + delta])
    bins = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (times >= left) & (times < right)
        bins.append((left, right, places.loc[mask].copy()))
    return bins


def bin_index_for_time(bins, when):
    stamp = parse_time(when)
    for i, (left, right, _) in enumerate(bins):
        if left <= stamp < right:
            return i
    if bins and stamp >= bins[-1][1]:
        return len(bins) - 1
    if bins and stamp < bins[0][0]:
        return 0
    raise ValueError('No time window contains {}'.format(when))


def _project(lon, lat, bounds, width, height):
    x = (float(lon) - bounds['lon0']) / (bounds['lon1'] - bounds['lon0'])
    y = (float(lat) - bounds['lat0']) / (bounds['lat1'] - bounds['lat0'])
    col = int(round(x * (width - 1)))
    row = int(round((1.0 - y) * (height - 1)))
    col = max(0, min(width - 1, col))
    row = max(0, min(height - 1, row))
    return row, col


def _cell_lonlat(row, col, bounds, width, height):
    x = (col + 0.5) / float(width)
    y = 1.0 - (row + 0.5) / float(height)
    lon = bounds['lon0'] + x * (bounds['lon1'] - bounds['lon0'])
    lat = bounds['lat0'] + y * (bounds['lat1'] - bounds['lat0'])
    return lon, lat


def render_world_land(width, height, bounds=None):
    """Scale the continent boxes onto an equirectangular grid."""
    bounds = bounds or WORLD_BOUNDS
    grid = []
    for r in range(height):
        row = []
        for c in range(width):
            lon, lat = _cell_lonlat(r, c, bounds, width, height)
            row.append(':' if is_land(lon, lat) else ' ')
        grid.append(row)
    return grid


def render_ascii(places, bounds=None, width=64, height=18, stations=None):
    """World map with probability-sized glyphs. North is up."""
    width = max(8, int(width))
    height = max(6, int(height))
    bounds = bounds or WORLD_BOUNDS
    grid = render_world_land(width, height, bounds=bounds)

    dots = aggregate_places(places)
    if stations is not None and not stations.empty and dots.empty:
        dots = aggregate_places(stations)

    for _, row in dots.iterrows():
        r, c = _project(
            row['meta_deploy_longitude'], row['meta_deploy_latitude'],
            bounds, width, height,
        )
        prob = row['rogue_prob'] if 'rogue_prob' in row.index else 0.0
        grid[r][c] = glyph_for_prob(prob)

    lines = [''.join(grid[r]) for r in range(height)]
    return '\n'.join(lines), bounds


def unique_stations(places):
    if places.empty:
        return places
    cols = ['meta_deploy_latitude', 'meta_deploy_longitude']
    extra = ['meta_station_name'] if 'meta_station_name' in places.columns else []
    return places[cols + extra].drop_duplicates().reset_index(drop=True)


def lon_ticks(width):
    """180W ... 0 ... 180E under the canvas."""
    if width < 20:
        return '180W' + ' ' * max(0, width - 8) + '180E'
    mid = '0'
    left = '180W'
    right = '180E'
    inner = width - len(left) - len(right)
    pad_left = max(0, inner // 2 - len(mid) // 2)
    pad_right = max(0, inner - pad_left - len(mid))
    return left + (' ' * pad_left) + mid + (' ' * pad_right) + right


def format_map(run_id, canvas, places, window_label, n_bins=None, bin_index=None):
    dots = aggregate_places(places) if places is not None and len(places) else places
    n = 0 if dots is None else len(dots)
    n_true = int((dots['is_rogue'] > 0).sum()) if n and 'is_rogue' in dots.columns else 0
    frame = ''
    if n_bins is not None and bin_index is not None:
        frame = '  frame {}/{}'.format(bin_index + 1, n_bins)
    header = [
        'World rogue map  {}  |  {}  |  {} places ({} rogues){}'.format(
            run_id, window_label, n, n_true, frame
        ),
        'Bigger symbol = higher P(rogue)    . <25%   o <50%   O <75%   @ >=75%',
    ]
    width = len(canvas.splitlines()[0]) if canvas else 40
    footer = [
        lon_ticks(width),
        'land :    map next | map prev | watch',
    ]
    return '\n'.join(header + [canvas] + footer)


def window_label(left, right=None, full=False):
    if full:
        return 'all time'
    return '{}  ->  {}'.format(left, right)


def state_path(out_dir, run_id):
    return os.path.join(store.run_dir(out_dir, run_id), STATE_NAME)


def load_map_state(out_dir, run_id):
    path = state_path(out_dir, run_id)
    if not os.path.isfile(path):
        return {'bin_index': 0, 'bin': '1h'}
    with open(path, 'r') as handle:
        return json.load(handle)


def save_map_state(out_dir, run_id, bin_index, bin_size):
    _, key = parse_bin(bin_size)
    payload = {
        'run_id': run_id,
        'bin_index': int(bin_index),
        'bin': key,
    }
    path = state_path(out_dir, run_id)
    tmp = path + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(tmp, path)
    return payload


def map_title(run_id, label, places, n_bins=None, bin_index=None):
    dots = aggregate_places(places) if places is not None and len(places) else places
    n = 0 if dots is None else len(dots)
    n_true = int((dots['is_rogue'] > 0).sum()) if n and 'is_rogue' in dots.columns else 0
    frame = ''
    if n_bins is not None and bin_index is not None:
        frame = '  |  frame {}/{}'.format(bin_index + 1, n_bins)
    return (
        'Rogue-wave risk  {}  |  {}  |  {} places ({} rogues){}'
        '\nLarger / warmer = higher P(rogue)'
    ).format(run_id, label, n, n_true, frame)


def resolve_frame(out_dir, run_id=None, min_prob=0.0, holdout_only=False,
                  actual_only=False, bin_size='1h', at_time=None, step=None):
    """Pick the places + title for one map view. Updates map_state for next/prev."""
    run_id, predictions = load_predictions(out_dir, run_id)
    places = flagged_places(
        predictions,
        min_prob=min_prob,
        holdout_only=holdout_only,
        actual_only=actual_only,
    )
    if places.empty:
        raise ValueError('No flagged places to map. Generate a report first.')

    bins = time_bins(places, bin_size=bin_size)
    state = load_map_state(out_dir, run_id)
    payload = {
        'run_id': run_id,
        'run_dir': store.run_dir(out_dir, run_id),
        'state': state,
        'n_bins': len(bins),
        'bin_index': None,
    }

    if at_time is None and step is None:
        payload['places'] = places
        payload['title'] = map_title(run_id, window_label(None, full=True), places)
        return payload

    if not bins:
        raise ValueError('No time windows in this run')

    if at_time is not None:
        index = bin_index_for_time(bins, at_time)
    else:
        index = int(state.get('bin_index', 0))
        has_state = os.path.isfile(state_path(out_dir, run_id))
        if step:
            if has_state:
                index += int(step)
            else:
                index = 0
        index = max(0, min(len(bins) - 1, index))

    left, right, frame = bins[index]
    payload['places'] = frame
    payload['bin_index'] = index
    payload['title'] = map_title(
        run_id, window_label(left, right), frame,
        n_bins=len(bins), bin_index=index,
    )
    payload['state'] = save_map_state(out_dir, run_id, index, bin_size)
    return payload


def iter_resolved_frames(out_dir, run_id=None, min_prob=0.0, holdout_only=False,
                         actual_only=False, bin_size='1h'):
    """Yield resolve_frame-like dicts for every time window."""
    run_id, predictions = load_predictions(out_dir, run_id)
    places = flagged_places(
        predictions,
        min_prob=min_prob,
        holdout_only=holdout_only,
        actual_only=actual_only,
    )
    if places.empty:
        raise ValueError('No flagged places to map. Generate a report first.')
    bins = time_bins(places, bin_size=bin_size)
    run_path = store.run_dir(out_dir, run_id)
    for index, (left, right, frame) in enumerate(bins):
        save_map_state(out_dir, run_id, index, bin_size)
        yield {
            'run_id': run_id,
            'run_dir': run_path,
            'places': frame,
            'bin_index': index,
            'n_bins': len(bins),
            'title': map_title(
                run_id, window_label(left, right), frame,
                n_bins=len(bins), bin_index=index,
            ),
        }


def build_view(out_dir, run_id=None, min_prob=0.0, holdout_only=False,
               actual_only=False, bin_size='1h', at_time=None, step=None,
               width=0, height=0):
    """Load predictions and return (run_id, text, state)."""
    run_id, predictions = load_predictions(out_dir, run_id)
    places = flagged_places(
        predictions,
        min_prob=min_prob,
        holdout_only=holdout_only,
        actual_only=actual_only,
    )
    if places.empty:
        raise ValueError('No flagged places to map. Generate a report first.')

    width, height = terminal_canvas_size(width, height)
    bins = time_bins(places, bin_size=bin_size)
    state = load_map_state(out_dir, run_id)

    if at_time is None and step is None:
        canvas, _ = render_ascii(places, bounds=WORLD_BOUNDS, width=width, height=height)
        text = format_map(run_id, canvas, places, window_label(None, full=True))
        return run_id, text, state

    if not bins:
        raise ValueError('No time windows in this run')

    if at_time is not None:
        index = bin_index_for_time(bins, at_time)
    else:
        index = int(state.get('bin_index', 0))
        has_state = os.path.isfile(state_path(out_dir, run_id))
        if step:
            if has_state:
                index += int(step)
            else:
                index = 0
        index = max(0, min(len(bins) - 1, index))

    left, right, frame = bins[index]
    canvas, _ = render_ascii(frame, bounds=WORLD_BOUNDS, width=width, height=height)
    text = format_map(
        run_id, canvas, frame, window_label(left, right),
        n_bins=len(bins), bin_index=index,
    )
    state = save_map_state(out_dir, run_id, index, bin_size)
    return run_id, text, state


def iter_watch_frames(out_dir, run_id=None, min_prob=0.0, holdout_only=False,
                      actual_only=False, bin_size='1h', width=0, height=0):
    """Yield (index, n_bins, text) for each time window."""
    run_id, predictions = load_predictions(out_dir, run_id)
    places = flagged_places(
        predictions,
        min_prob=min_prob,
        holdout_only=holdout_only,
        actual_only=actual_only,
    )
    if places.empty:
        raise ValueError('No flagged places to map. Generate a report first.')
    width, height = terminal_canvas_size(width, height)
    bins = time_bins(places, bin_size=bin_size)
    for index, (left, right, frame) in enumerate(bins):
        canvas, _ = render_ascii(
            frame, bounds=WORLD_BOUNDS, width=width, height=height
        )
        text = format_map(
            run_id, canvas, frame, window_label(left, right),
            n_bins=len(bins), bin_index=index,
        )
        save_map_state(out_dir, run_id, index, bin_size)
        yield index, len(bins), text
