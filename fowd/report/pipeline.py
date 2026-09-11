"""
pipeline.py

Process → QC → featurize → train → evaluate → score places → report.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from .features import (
    add_labels,
    apply_qc,
    feature_columns,
    hs_column,
    load_columns,
    normalize_interval,
)
from . import store
from .progress import (
    NUM_BOOST_ROUNDS,
    BoostProgress,
    CheckpointCallback,
    stage_line,
    training_bar,
)
from .synthetic import generate_catalogue


class PipelineError(Exception):
    """A stage check failed; do not invent metrics."""


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_artifact(run_dir, name, payload):
    path = os.path.join(run_dir, 'artifacts', '{}.json'.format(name))
    with open(path, 'w') as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write('\n')
    return path


def _as_str(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    text = str(value)
    if text.startswith("b'") and text.endswith("'"):
        return text[2:-1]
    return text.strip()


def load_fowd_netcdf(path, interval='30m'):
    """Load selected FOWD variables into a DataFrame (no raw elevation)."""
    import xarray as xr

    wanted = load_columns(interval)
    with xr.open_dataset(path) as dataset:
        station = None
        if 'meta_station_name' in dataset:
            raw = np.asarray(dataset['meta_station_name'].values)
            station = _as_str(raw.flat[0] if raw.size else raw)

        data = {}
        for name in wanted:
            if name == 'meta_station_name':
                continue
            if name not in dataset.variables and name not in dataset.coords:
                continue
            arr = np.asarray(dataset[name].values)
            if arr.ndim > 1:
                continue
            data[name] = arr

        frame = pd.DataFrame(data)
        if station is not None:
            frame['meta_station_name'] = station
        elif 'meta_station_name' not in frame.columns:
            frame['meta_station_name'] = os.path.splitext(os.path.basename(path))[0]
    return frame


def stage_process(input_files=None, cdip_folder=None, generic_infile=None,
                  synthetic=False, run_dir=None):
    """Resolve source catalogues. Processing writes netCDF under the run folder."""
    input_files = list(input_files or [])
    n_sources = sum([
        bool(input_files),
        bool(cdip_folder),
        bool(generic_infile),
        bool(synthetic),
    ])
    if n_sources == 0:
        synthetic = True
        n_sources = 1
    if n_sources > 1:
        raise PipelineError('Choose one source: --input, --cdip-folder, --generic-infile, or --synthetic')

    if synthetic:
        return {
            'source': 'synthetic',
            'files': [],
            'ok': True,
        }

    produced = []
    if input_files:
        for path in input_files:
            if not os.path.isfile(path):
                raise PipelineError('Input file not found: {}'.format(path))
            produced.append(os.path.abspath(path))
        return {'source': 'input', 'files': produced, 'ok': True}

    dest = os.path.join(run_dir, 'processed')
    os.makedirs(dest, exist_ok=True)

    if cdip_folder:
        from ..cdip import process_cdip_station
        process_cdip_station(cdip_folder, dest)
        source = 'cdip'
    else:
        from ..generic_source import process_file
        process_file(generic_infile, dest)
        source = 'generic'

    produced = sorted(glob.glob(os.path.join(dest, '*.nc')))
    return {
        'source': source,
        'files': produced,
        'ok': bool(produced),
    }


def check_process(artifact):
    if artifact.get('source') == 'synthetic':
        return
    if not artifact.get('files'):
        raise PipelineError('Processing produced no FOWD netCDF files')


def stage_load_qc(process_artifact, interval='30m', seed=0, n_waves=8000):
    interval = normalize_interval(interval)
    if process_artifact.get('source') == 'synthetic':
        frame = generate_catalogue(n_waves=n_waves, seed=seed, interval=interval)
        dropped = 0
        source = 'synthetic'
    else:
        frames = [load_fowd_netcdf(path, interval=interval) for path in process_artifact['files']]
        if not frames:
            raise PipelineError('No catalogues to load')
        frame = pd.concat(frames, ignore_index=True)
        frame, dropped = apply_qc(frame, interval=interval)
        source = process_artifact.get('source')

    if frame.empty:
        raise PipelineError('No rows left after QC')

    frame = add_labels(frame, interval=interval)
    return frame, {
        'source': source,
        'n_rows': int(len(frame)),
        'n_dropped': int(dropped),
        'n_rogues': int(frame['is_rogue'].sum()),
        'interval': interval,
        'ok': True,
    }


def check_load_qc(artifact):
    if artifact.get('n_rows', 0) < 50:
        raise PipelineError('Need at least 50 QC-passed waves, got {}'.format(artifact.get('n_rows')))
    required = ('n_rogues',)
    for key in required:
        if key not in artifact:
            raise PipelineError('Load artifact missing {}'.format(key))


def stage_featurize(frame, interval='30m'):
    interval = normalize_interval(interval)
    available = [col for col in feature_columns(interval) if col in frame.columns]
    if not available:
        raise PipelineError('None of the expected sea-state features are present')

    work = frame.dropna(subset=available + ['is_rogue']).copy()
    if work.empty:
        raise PipelineError('All rows have missing features')

    return work, available, {
        'n_rows': int(len(work)),
        'n_features': int(len(available)),
        'features': available,
        'n_rogues': int(work['is_rogue'].sum()),
        'rogue_rate': float(work['is_rogue'].mean()),
        'ok': True,
    }


def check_featurize(artifact):
    if artifact.get('n_rogues', 0) < 5:
        raise PipelineError(
            'Need at least 5 rogue waves in the labelled table, got {}'.format(
                artifact.get('n_rogues')
            )
        )
    if artifact.get('n_features', 0) < 3:
        raise PipelineError('Too few usable features')


def time_split_indices(frame, train_frac=0.8):
    """Chronological split, done separately inside each station."""
    train_pos = []
    test_pos = []
    if 'meta_station_name' in frame.columns:
        grouped = frame.groupby(frame['meta_station_name'], sort=False)
    else:
        grouped = [(None, frame)]

    for _, part in grouped:
        if 'wave_start_time' in part.columns:
            part = part.sort_values('wave_start_time', kind='mergesort')
        n_part = len(part)
        cut = int(round(n_part * train_frac))
        cut = min(max(cut, 1), n_part - 1) if n_part > 1 else n_part
        train_pos.extend(part.index[:cut].tolist())
        test_pos.extend(part.index[cut:].tolist())

    return train_pos, test_pos


def _sklearn_metrics():
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as exc:
        raise PipelineError(
            'The report pipeline needs scikit-learn. '
            'Install with: pip install ".[pipeline]"'
        ) from exc
    return average_precision_score, roc_auc_score


def _try_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except (ImportError, OSError):
        return None


class _SklearnBoost:
    """Thin wrapper so sklearn and LightGBM share predict / importance / save."""

    def __init__(self, estimator):
        self.estimator = estimator

    def predict(self, frame):
        return self.estimator.predict_proba(frame)[:, 1]

    def feature_importance(self, importance_type='gain'):
        return np.asarray(self.estimator.feature_importances_, dtype=float)

    def save_model(self, path):
        import pickle
        with open(path, 'wb') as handle:
            pickle.dump(self.estimator, handle)


def _checkpoint_dir(run_dir):
    path = os.path.join(run_dir, 'checkpoints')
    os.makedirs(path, exist_ok=True)
    return path


def _save_sklearn_checkpoint(estimator, directory, round_i):
    import pickle
    path = os.path.join(directory, 'round_{:06d}.pkl'.format(round_i))
    with open(path, 'wb') as handle:
        pickle.dump(estimator, handle)
    return path


def _fit_model(x_train, y_train, scale_pos_weight, run_dir, rounds=None,
               save_every=None):
    n_rows = int(len(x_train))
    n_rounds = int(rounds or NUM_BOOST_ROUNDS)
    n_save = int(save_every) if save_every else 0
    if n_rounds < 1:
        raise PipelineError('--rounds must be at least 1')
    ckpt_dir = _checkpoint_dir(run_dir) if n_save > 0 else None
    checkpoints = []

    lgb = _try_lightgbm()
    if lgb is not None:
        dataset = lgb.Dataset(x_train, label=y_train)
        params = {
            'objective': 'binary',
            'metric': 'average_precision',
            'verbosity': -1,
            'scale_pos_weight': float(scale_pos_weight),
            'num_leaves': 31,
            'learning_rate': 0.05,
            'min_child_samples': 20,
            'feature_pre_filter': False,
        }
        tracker = BoostProgress(
            total=n_rounds,
            desc='Training LightGBM ({} waves)'.format(n_rows),
        )
        saver = CheckpointCallback(ckpt_dir, n_save, extension='txt') if ckpt_dir else None
        callbacks = [tracker]
        if saver is not None:
            callbacks.append(saver)
        try:
            model = lgb.train(
                params,
                dataset,
                num_boost_round=n_rounds,
                valid_sets=[dataset],
                valid_names=['train'],
                callbacks=callbacks,
            )
        finally:
            tracker.close()
        if saver is not None:
            checkpoints = list(saver.saved)
        model_path = os.path.join(run_dir, 'model.txt')
        model.save_model(model_path)
        return model, model_path, 'lightgbm', checkpoints

    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError as exc:
        raise PipelineError(
            'Need lightgbm or scikit-learn. Install with: pip install ".[pipeline]"'
        ) from exc

    weights = np.where(np.asarray(y_train) == 1, float(scale_pos_weight), 1.0)
    estimator = GradientBoostingClassifier(
        n_estimators=1,
        max_depth=3,
        learning_rate=0.05,
        random_state=0,
        warm_start=True,
    )
    bar = training_bar(
        total=n_rounds,
        desc='Training GBT ({} waves)'.format(n_rows),
        unit='tree',
    )
    try:
        for round_i in range(1, n_rounds + 1):
            estimator.n_estimators = round_i
            estimator.fit(x_train, y_train, sample_weight=weights)
            if ckpt_dir is not None and n_save > 0 and round_i % n_save == 0:
                checkpoints.append(_save_sklearn_checkpoint(estimator, ckpt_dir, round_i))
            bar.update(1)
    finally:
        bar.close()
    model = _SklearnBoost(estimator)
    model_path = os.path.join(run_dir, 'model.pkl')
    model.save_model(model_path)
    return model, model_path, 'sklearn_gbt', checkpoints


def stage_train(frame, feature_names, run_dir, rounds=None, save_every=None):
    _sklearn_metrics()

    train_idx, test_idx = time_split_indices(frame)
    if not test_idx:
        raise PipelineError('Time split produced an empty hold-out set')

    split = pd.Series('train', index=frame.index)
    split.loc[test_idx] = 'test'
    frame = frame.copy()
    frame['split'] = split

    y = frame['is_rogue'].astype(int)
    if int(y.loc[test_idx].sum()) < 1:
        raise PipelineError('Hold-out set has no rogue waves; cannot compute PR-AUC')

    x_train = frame.loc[train_idx, feature_names]
    y_train = y.loc[train_idx]
    n_pos = max(int(y_train.sum()), 1)
    n_neg = max(int((y_train == 0).sum()), 1)

    model, model_path, backend, checkpoints = _fit_model(
        x_train, y_train, float(n_neg) / float(n_pos), run_dir,
        rounds=rounds, save_every=save_every,
    )

    return model, frame, {
        'n_train': int(len(train_idx)),
        'n_test': int(len(test_idx)),
        'n_train_rogues': int(y.loc[train_idx].sum()),
        'n_test_rogues': int(y.loc[test_idx].sum()),
        'model_path': model_path,
        'model_backend': backend,
        'n_rounds': int(rounds or NUM_BOOST_ROUNDS),
        'save_every': int(save_every) if save_every else 0,
        'checkpoints': checkpoints,
        'split': 'time_per_station',
        'ok': True,
    }


def check_train(artifact):
    if artifact.get('split') != 'time_per_station':
        raise PipelineError('Official score must use a time-per-station split')
    if not artifact.get('model_path') or not os.path.isfile(artifact['model_path']):
        raise PipelineError('Model file was not written')


def stage_evaluate(model, frame, feature_names, run_dir):
    average_precision_score, roc_auc_score = _sklearn_metrics()

    test = frame.loc[frame['split'] == 'test']
    y_true = test['is_rogue'].astype(int).values
    y_prob = np.asarray(model.predict(test[feature_names]), dtype=float)
    pr_auc = float(average_precision_score(y_true, y_prob))
    try:
        roc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc = float('nan')
    baseline = float(y_true.mean())

    gain = np.asarray(model.feature_importance(importance_type='gain'), dtype=float)
    importances = pd.DataFrame({
        'feature': feature_names,
        'gain': gain,
    }).sort_values('gain', ascending=False)
    imp_path = os.path.join(run_dir, 'importances.csv')
    importances.to_csv(imp_path, index=False)

    return {
        'pr_auc': pr_auc,
        'roc_auc': roc,
        'baseline_pr_auc': baseline,
        'pr_auc_lift': pr_auc - baseline,
        'importances_path': imp_path,
        'top_features': importances.head(8).to_dict(orient='records'),
        'ok': True,
    }


def check_evaluate(artifact):
    if 'pr_auc' not in artifact:
        raise PipelineError('PR-AUC was not computed')
    if np.isnan(artifact.get('pr_auc', np.nan)):
        raise PipelineError('PR-AUC is not a number')


def stage_score_places(model, frame, feature_names, interval, run_dir):
    interval = normalize_interval(interval)
    scored = frame.copy()
    scored['rogue_prob'] = np.asarray(model.predict(scored[feature_names]), dtype=float)

    keep = [
        'meta_station_name',
        'wave_start_time',
        'wave_end_time',
        'meta_deploy_latitude',
        'meta_deploy_longitude',
        'meta_water_depth',
        'wave_height',
        'relative_wave_height',
        'is_rogue',
        'rogue_prob',
        'split',
        hs_column(interval),
    ]
    extra = [name for name in feature_names if name not in keep]
    columns = [name for name in keep + extra if name in scored.columns]
    table = scored[columns].sort_values('rogue_prob', ascending=False)
    pred_path = os.path.join(run_dir, 'predictions.csv')
    table.to_csv(pred_path, index=False)

    preview = table.head(5)
    return table, {
        'predictions_path': pred_path,
        'n_scored': int(len(table)),
        'max_rogue_prob': float(table['rogue_prob'].max()) if len(table) else 0.0,
        'preview': _preview_rows(preview),
        'ok': True,
    }


def _preview_rows(frame):
    rows = []
    for _, row in frame.iterrows():
        item = {}
        for key, value in row.items():
            if hasattr(value, 'isoformat'):
                item[key] = str(value)
            else:
                item[key] = value
        rows.append(item)
    return rows


def stage_propose(eval_artifact, interval='30m'):
    interval = normalize_interval(interval)
    suggestions = []
    lift = eval_artifact.get('pr_auc_lift', 0.0)
    top = eval_artifact.get('top_features') or []
    top_name = top[0]['feature'] if top else ''

    if lift < 0.02:
        other = '10m' if interval == '30m' else '30m'
        suggestions.append(
            'PR-AUC is close to the dummy baseline; retry with --sea-state-interval {}'.format(other)
        )
    if 'crest_trough_correlation' in top_name:
        suggestions.append(
            'Crest-trough correlation dominates; bin the next run by water depth'
        )
    if 'benjamin_feir' in top_name:
        suggestions.append(
            'BFI ranks high here; compare against a model that drops BFI to test robustness'
        )
    if not suggestions:
        suggestions.append(
            'Hold-out PR-AUC beats baseline; inspect `fowd report filter` for high-risk places'
        )
    return {'suggestions': suggestions, 'ok': True}


def write_failure_report(run_dir, stage, error):
    payload = {
        'status': 'failed',
        'failed_stage': stage,
        'error': str(error),
    }
    report_json = os.path.join(run_dir, 'report.json')
    with open(report_json, 'w') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')
    report_md = os.path.join(run_dir, 'report.md')
    with open(report_md, 'w') as handle:
        handle.write('# FOWD discovery report (failed)\n\n')
        handle.write('Stopped at **{}**: {}\n'.format(stage, error))
    return payload


def write_success_report(run_dir, artifacts):
    eval_art = artifacts['evaluate']
    score_art = artifacts['score_places']
    payload = {
        'status': 'ok',
        'source': artifacts['process'].get('source'),
        'interval': artifacts['load_qc'].get('interval'),
        'n_rows': artifacts['featurize'].get('n_rows'),
        'n_rogues': artifacts['featurize'].get('n_rogues'),
        'rogue_rate': artifacts['featurize'].get('rogue_rate'),
        'n_train': artifacts['train'].get('n_train'),
        'n_test': artifacts['train'].get('n_test'),
        'pr_auc': eval_art.get('pr_auc'),
        'roc_auc': eval_art.get('roc_auc'),
        'baseline_pr_auc': eval_art.get('baseline_pr_auc'),
        'pr_auc_lift': eval_art.get('pr_auc_lift'),
        'top_features': eval_art.get('top_features'),
        'suggestions': artifacts['propose'].get('suggestions'),
        'highest_risk_places': score_art.get('preview'),
    }
    report_json = os.path.join(run_dir, 'report.json')
    with open(report_json, 'w') as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write('\n')

    lines = [
        '# FOWD discovery report',
        '',
        '- Status: ok',
        '- Source: {}'.format(payload['source']),
        '- Sea-state interval: {}'.format(payload['interval']),
        '- Waves: {} ({} rogues, rate {:.4f})'.format(
            payload['n_rows'], payload['n_rogues'], payload['rogue_rate']
        ),
        '- Split: time per station (train {}, test {})'.format(
            payload['n_train'], payload['n_test']
        ),
        '- Hold-out PR-AUC: {:.4f} (dummy baseline {:.4f}, lift {:.4f})'.format(
            payload['pr_auc'], payload['baseline_pr_auc'], payload['pr_auc_lift']
        ),
        '- Hold-out ROC-AUC: {:.4f}'.format(payload['roc_auc']),
        '',
        '## Top features (gain)',
        '',
    ]
    for row in payload.get('top_features') or []:
        lines.append('- {}: {:.4f}'.format(row['feature'], row['gain']))
    lines.extend(['', '## Highest-risk places (preview)', ''])
    for row in payload.get('highest_risk_places') or []:
        lines.append(
            '- {station} t={time} lat={lat} lon={lon} P(rogue)={prob:.3f} '
            'actual={actual}'.format(
                station=row.get('meta_station_name'),
                time=row.get('wave_start_time'),
                lat=row.get('meta_deploy_latitude'),
                lon=row.get('meta_deploy_longitude'),
                prob=float(row.get('rogue_prob', 0)),
                actual=row.get('is_rogue'),
            )
        )
    lines.extend(['', '## Next experiments', ''])
    for idea in payload.get('suggestions') or []:
        lines.append('- {}'.format(idea))
    lines.append('')

    report_md = os.path.join(run_dir, 'report.md')
    with open(report_md, 'w') as handle:
        handle.write('\n'.join(lines))
    return payload


def generate_report(out_dir, input_files=None, cdip_folder=None, generic_infile=None,
                    synthetic=False, interval='30m', seed=0, rounds=None, n_waves=8000,
                    save_every=None):
    """Run the full discovery pipeline into a new run directory."""
    interval = normalize_interval(interval)
    if not input_files and not cdip_folder and not generic_infile:
        synthetic = True
    source = 'synthetic' if synthetic else (
        'cdip' if cdip_folder else ('generic' if generic_infile else 'input')
    )
    run_id, run_path = store.create_run(out_dir, source=source)
    artifacts = {}

    stages = [
        ('process', lambda: stage_process(
            input_files=input_files,
            cdip_folder=cdip_folder,
            generic_infile=generic_infile,
            synthetic=synthetic,
            run_dir=run_path,
        ), check_process),
    ]

    n_stages = 7
    try:
        stage_line(1, n_stages, 'process source')
        process_art = stages[0][1]()
        check_process(process_art)
        write_artifact(run_path, 'process', process_art)
        artifacts['process'] = process_art

        stage_line(2, n_stages, 'load + QC')
        frame, load_art = stage_load_qc(
            process_art, interval=interval, seed=seed, n_waves=n_waves
        )
        check_load_qc(load_art)
        write_artifact(run_path, 'load_qc', load_art)
        artifacts['load_qc'] = load_art

        stage_line(3, n_stages, 'featurize')
        frame, feature_names, feat_art = stage_featurize(frame, interval=interval)
        check_featurize(feat_art)
        write_artifact(run_path, 'featurize', feat_art)
        artifacts['featurize'] = feat_art

        stage_line(4, n_stages, 'train model')
        model, frame, train_art = stage_train(
            frame, feature_names, run_path, rounds=rounds, save_every=save_every
        )
        check_train(train_art)
        write_artifact(run_path, 'train', train_art)
        artifacts['train'] = train_art

        stage_line(5, n_stages, 'evaluate hold-out')
        eval_art = stage_evaluate(model, frame, feature_names, run_path)
        check_evaluate(eval_art)
        write_artifact(run_path, 'evaluate', eval_art)
        artifacts['evaluate'] = eval_art

        stage_line(6, n_stages, 'score places')
        _, score_art = stage_score_places(model, frame, feature_names, interval, run_path)
        write_artifact(run_path, 'score_places', score_art)
        artifacts['score_places'] = score_art

        stage_line(7, n_stages, 'write report')
        propose_art = stage_propose(eval_art, interval=interval)
        write_artifact(run_path, 'propose', propose_art)
        artifacts['propose'] = propose_art

        report = write_success_report(run_path, artifacts)
        store.update_run(
            out_dir, run_id,
            status='ok',
            pr_auc=report.get('pr_auc'),
            n_rows=report.get('n_rows'),
        )
        return {
            'run_id': run_id,
            'run_dir': run_path,
            'report': report,
        }
    except Exception as exc:
        stage_name = 'unknown'
        for name in ('propose', 'score_places', 'evaluate', 'train', 'featurize',
                     'load_qc', 'process'):
            if name not in artifacts:
                stage_name = name
        write_failure_report(run_path, stage_name, exc)
        store.update_run(out_dir, run_id, status='failed', error=str(exc))
        if isinstance(exc, PipelineError):
            raise
        raise PipelineError('{} failed: {}'.format(stage_name, exc)) from exc
