"""
store.py

On-disk run folders and an index of discovery reports.
"""

import json
import os
import uuid
from datetime import datetime

INDEX_NAME = 'index.json'


def runs_root(out_dir):
    return os.path.join(out_dir, 'runs')


def index_path(out_dir):
    return os.path.join(out_dir, INDEX_NAME)


def read_index(out_dir):
    path = index_path(out_dir)
    if not os.path.isfile(path):
        return {'runs': []}
    with open(path, 'r') as handle:
        return json.load(handle)


def write_index(out_dir, index):
    os.makedirs(out_dir, exist_ok=True)
    path = index_path(out_dir)
    tmp = path + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(tmp, path)


def create_run(out_dir, source='unknown'):
    """Create a new run directory and register it. Returns (run_id, run_dir)."""
    stamp = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
    run_id = '{}_{}'.format(stamp, uuid.uuid4().hex[:6])
    run_dir = os.path.join(runs_root(out_dir), run_id)
    os.makedirs(os.path.join(run_dir, 'artifacts'), exist_ok=True)

    index = read_index(out_dir)
    index.setdefault('runs', []).append({
        'id': run_id,
        'created': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'status': 'running',
        'source': source,
        'path': os.path.join('runs', run_id),
    })
    write_index(out_dir, index)
    return run_id, run_dir


def update_run(out_dir, run_id, **fields):
    index = read_index(out_dir)
    for row in index.get('runs', []):
        if row.get('id') == run_id:
            row.update(fields)
            break
    write_index(out_dir, index)


def list_runs(out_dir):
    return list(read_index(out_dir).get('runs', []))


def run_dir(out_dir, run_id):
    return os.path.join(runs_root(out_dir), run_id)


def resolve_run_id(out_dir, run_id=None):
    """Return a run id, defaulting to the newest run that has predictions."""
    runs = list_runs(out_dir)
    if not runs:
        raise FileNotFoundError('No reports in {}. Run `fowd report generate` first.'.format(out_dir))
    if run_id is None:
        for row in reversed(runs):
            pred = os.path.join(run_dir(out_dir, row['id']), 'predictions.csv')
            if os.path.isfile(pred):
                return row['id']
        return runs[-1]['id']
    known = {row['id'] for row in runs}
    if run_id not in known:
        raise FileNotFoundError('Unknown run {!r} in {}'.format(run_id, out_dir))
    return run_id
