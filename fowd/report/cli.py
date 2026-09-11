"""
cli.py

Click commands for discovery reports.
"""

import os

import click

from . import store
from .features import normalize_interval
from .filter import filter_rogue_places, format_places, load_predictions
from .maprender import flagged_places, resolve_frame
from .pipeline import PipelineError, generate_report
from .worldmap import MapViewer, interactive_world_map


def _run_generate(input_files, cdip_folder, generic_infile, synthetic, out_folder,
                  sea_state_interval, seed, rounds, n_waves, save_every):
    os.makedirs(out_folder, exist_ok=True)
    try:
        result = generate_report(
            out_folder,
            input_files=input_files,
            cdip_folder=cdip_folder,
            generic_infile=generic_infile,
            synthetic=synthetic,
            interval=normalize_interval(sea_state_interval),
            seed=seed,
            rounds=rounds,
            n_waves=n_waves,
            save_every=save_every,
        )
    except PipelineError as exc:
        raise click.ClickException(str(exc))

    rec = result['report']
    click.echo('Wrote report {}'.format(result['run_id']))
    click.echo('  directory: {}'.format(result['run_dir']))
    if rec.get('status') == 'ok':
        click.echo('  hold-out PR-AUC: {:.4f} (baseline {:.4f})'.format(
            rec['pr_auc'], rec['baseline_pr_auc']
        ))
        click.echo('  filter high-risk places with: fowd report filter -o {}'.format(out_folder))
    return result


@click.group('report')
def report():
    """Generate and filter rogue-wave discovery reports."""


@report.command('generate')
@click.option(
    '--input', 'input_files',
    type=click.Path(dir_okay=False, exists=True, readable=True),
    multiple=True,
    help='Existing FOWD netCDF catalogue(s).',
)
@click.option(
    '--cdip-folder',
    type=click.Path(file_okay=False, exists=True, readable=True),
    help='Raw CDIP station folder to process first.',
)
@click.option(
    '--generic-infile',
    type=click.Path(dir_okay=False, exists=True, readable=True),
    help='Generic netCDF (time, displacement) to process first.',
)
@click.option('--synthetic', is_flag=True, help='Use a built-in synthetic catalogue.')
@click.option(
    '-o', '--out-folder',
    type=click.Path(file_okay=False, writable=True),
    default='fowd-reports',
    show_default=True,
)
@click.option(
    '--sea-state-interval',
    default='30m',
    show_default=True,
    help='Sea-state window: 10m, 30m, or dynamic.',
)
@click.option('--seed', default=0, show_default=True, type=int)
@click.option(
    '--rounds',
    default=180,
    show_default=True,
    type=int,
    help='Boosting rounds. Raise this to train longer.',
)
@click.option(
    '--n-waves',
    default=8000,
    show_default=True,
    type=int,
    help='Synthetic catalogue size (ignored for real netCDF / CDIP).',
)
@click.option(
    '--save-every',
    default=0,
    show_default=True,
    type=int,
    help='Write a checkpoint every N rounds (0 = only save the final model).',
)
def generate(input_files, cdip_folder, generic_infile, synthetic, out_folder,
             sea_state_interval, seed, rounds, n_waves, save_every):
    """Train a model and write a discovery report."""
    _run_generate(
        input_files, cdip_folder, generic_infile, synthetic, out_folder,
        sea_state_interval, seed, rounds, n_waves, save_every,
    )


@report.command('train')
@click.option(
    '--input', 'input_files',
    type=click.Path(dir_okay=False, exists=True, readable=True),
    multiple=True,
    help='Existing FOWD netCDF catalogue(s).',
)
@click.option(
    '--cdip-folder',
    type=click.Path(file_okay=False, exists=True, readable=True),
    help='Raw CDIP station folder to process first.',
)
@click.option(
    '--generic-infile',
    type=click.Path(dir_okay=False, exists=True, readable=True),
    help='Generic netCDF (time, displacement) to process first.',
)
@click.option('--synthetic', is_flag=True, help='Use a built-in synthetic catalogue.')
@click.option(
    '-o', '--out-folder',
    type=click.Path(file_okay=False, writable=True),
    default='fowd-reports',
    show_default=True,
)
@click.option(
    '--sea-state-interval',
    default='30m',
    show_default=True,
    help='Sea-state window: 10m, 30m, or dynamic.',
)
@click.option('--seed', default=0, show_default=True, type=int)
@click.option(
    '--rounds',
    default=5000,
    show_default=True,
    type=int,
    help='Boosting rounds for a long train.',
)
@click.option(
    '--n-waves',
    default=8000,
    show_default=True,
    type=int,
    help='Synthetic catalogue size (ignored for real netCDF / CDIP).',
)
@click.option(
    '--save-every',
    default=1000,
    show_default=True,
    type=int,
    help='Write a checkpoint every N rounds.',
)
def train(input_files, cdip_folder, generic_infile, synthetic, out_folder,
          sea_state_interval, seed, rounds, n_waves, save_every):
    """Long training run. Saves a checkpoint every --save-every rounds."""
    _run_generate(
        input_files, cdip_folder, generic_infile, synthetic, out_folder,
        sea_state_interval, seed, rounds, n_waves, save_every,
    )


@report.command('list')
@click.option(
    '-o', '--out-folder',
    type=click.Path(file_okay=False),
    default='fowd-reports',
    show_default=True,
)
def list_reports(out_folder):
    """List generated discovery runs."""
    runs = store.list_runs(out_folder)
    if not runs:
        click.echo('No reports in {}'.format(out_folder))
        return
    for row in runs:
        extra = ''
        if row.get('pr_auc') is not None:
            extra = '  PR-AUC={:.4f}'.format(row['pr_auc'])
        click.echo('{id}  {status}  {source}{extra}'.format(
            id=row.get('id'),
            status=row.get('status'),
            source=row.get('source'),
            extra=extra,
        ))


@report.command('filter')
@click.option('--run', 'run_id', default=None, help='Run id [default: latest].')
@click.option(
    '-o', '--out-folder',
    type=click.Path(file_okay=False),
    default='fowd-reports',
    show_default=True,
)
@click.option('--top', default=20, show_default=True, type=int,
              help='How many high-risk places to show.')
@click.option('--min-prob', default=0.0, show_default=True, type=float,
              help='Drop places below this P(rogue).')
@click.option('--holdout-only', is_flag=True,
              help='Only score the time hold-out (not the training window).')
@click.option(
    '--most-distinct',
    is_flag=True,
    help='Keep the places most likely to have a rogue wave (highest P(rogue)).',
)
@click.option(
    '--actual',
    is_flag=True,
    help='Show only confirmed rogues (wave height / Hs >= 2).',
)
@click.option(
    '--csv', 'csv_path',
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help='Optional path to write the filtered rows.',
)
def filter_cmd(run_id, out_folder, top, min_prob, holdout_only, most_distinct,
               actual, csv_path):
    """List places that look like they would have a rogue wave.

    Ranked by the trained model's P(rogue). --most-distinct is the same ranking
    (highest predicted rogue-wave risk). --actual lists measured rogues only.
    """
    del most_distinct
    try:
        run_id, predictions = load_predictions(out_folder, run_id)
        table, summary = filter_rogue_places(
            predictions,
            top=top,
            min_prob=min_prob,
            holdout_only=holdout_only,
            actual_only=actual,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))

    click.echo(format_places(table, summary, run_id))
    if csv_path:
        table.to_csv(csv_path, index=False)
        click.echo('')
        click.echo('Wrote {}'.format(csv_path))


@report.command('show')
@click.argument('run_id')
@click.option(
    '-o', '--out-folder',
    type=click.Path(file_okay=False),
    default='fowd-reports',
    show_default=True,
)
def show(run_id, out_folder):
    """Print metrics and importances for one run."""
    import json

    run_id = store.resolve_run_id(out_folder, run_id)
    run_path = store.run_dir(out_folder, run_id)
    report_path = os.path.join(run_path, 'report.json')
    if not os.path.isfile(report_path):
        raise click.ClickException('No report.json for {}'.format(run_id))

    with open(report_path, 'r') as handle:
        payload = json.load(handle)

    click.echo('Run {}'.format(run_id))
    click.echo('  status: {}'.format(payload.get('status')))
    if payload.get('status') != 'ok':
        click.echo('  error: {}'.format(payload.get('error')))
        return

    click.echo('  source: {}'.format(payload.get('source')))
    click.echo('  interval: {}'.format(payload.get('interval')))
    click.echo('  waves: {} ({} rogues)'.format(payload.get('n_rows'), payload.get('n_rogues')))
    click.echo('  PR-AUC: {:.4f} (baseline {:.4f})'.format(
        payload.get('pr_auc', float('nan')),
        payload.get('baseline_pr_auc', float('nan')),
    ))
    click.echo('  ROC-AUC: {:.4f}'.format(payload.get('roc_auc', float('nan'))))
    click.echo('  top features:')
    for row in payload.get('top_features') or []:
        click.echo('    {}: {:.4f}'.format(row['feature'], row['gain']))
    click.echo('  next experiments:')
    for idea in payload.get('suggestions') or []:
        click.echo('    - {}'.format(idea))


def _map_kwargs(run_id, out_folder, min_prob, holdout_only, actual, bin_size):
    return dict(
        out_dir=out_folder,
        run_id=run_id,
        min_prob=min_prob,
        holdout_only=holdout_only,
        actual_only=actual,
        bin_size=bin_size,
    )


def _png_path(out_folder, run_id):
    return os.path.join(store.run_dir(out_folder, run_id), 'world_map.png')


def _draw_frame(out_folder, run_id, min_prob, holdout_only, actual, bin_size,
                at_time=None, step=None, show=True):
    del holdout_only  # collected and projected share one map
    try:
        run_id, predictions = load_predictions(out_folder, run_id)
        png_path = _png_path(out_folder, run_id)
        if step is not None or at_time is not None:
            frame = resolve_frame(
                out_folder, run_id=run_id, min_prob=min_prob,
                holdout_only=False, actual_only=actual, bin_size=bin_size,
                at_time=at_time, step=step,
            )
            viewer = MapViewer(
                flagged_places(predictions, min_prob=min_prob, actual_only=actual),
                frame['run_id'], png_path, bin_size=bin_size, show=show,
            )
            viewer.set_index(frame.get('bin_index') or 0, save=True)
            import matplotlib.pyplot as plt
            if show:
                plt.show()
            else:
                plt.close(viewer.fig)
            title = frame.get('title') or ''
            click.echo(title.split('\n')[0] if title else '')
            click.echo('Wrote {}'.format(png_path))
            return frame

        interactive_world_map(
            predictions, run_id, png_path, show=show, bin_size=bin_size,
            min_prob=min_prob, actual_only=actual,
        )
        click.echo('Wrote {}'.format(png_path))
        return png_path
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc))


@report.group('map', invoke_without_command=True)
@click.pass_context
@click.option('--run', 'run_id', default=None, help='Run id [default: latest].')
@click.option(
    '-o', '--out-folder',
    type=click.Path(file_okay=False),
    default='fowd-reports',
    show_default=True,
)
@click.option('--at', 'at_time', default=None, help='Show the window that contains this time.')
@click.option('--bin', 'bin_size', default='1h', show_default=True,
              help='Time window size: 30min, 1h, or 6h.')
@click.option('--min-prob', default=0.0, show_default=True, type=float)
@click.option('--holdout-only', is_flag=True)
@click.option('--actual', is_flag=True, help='Map confirmed rogues only.')
@click.option('--show/--no-show', default=True, show_default=True,
              help='Open a matplotlib window (PNG is always saved).')
def map_cmd(ctx, run_id, out_folder, at_time, bin_size, min_prob, holdout_only,
            actual, show):
    """Interactive cartopy world map with a time slider (zoom/pan in the window)."""
    ctx.ensure_object(dict)
    ctx.obj.update(_map_kwargs(
        run_id, out_folder, min_prob, holdout_only, actual, bin_size
    ))
    ctx.obj['show'] = show
    ctx.obj['at_time'] = at_time
    if ctx.invoked_subcommand is not None:
        return
    _draw_frame(
        out_folder, run_id, min_prob, holdout_only, actual, bin_size,
        at_time=at_time, show=show,
    )


def _step_from_ctx(ctx, step, out_folder, run_id):
    kw = dict(ctx.obj)
    _draw_frame(step=step, at_time=None, **{
        'out_folder': out_folder or kw['out_dir'],
        'run_id': run_id if run_id is not None else kw['run_id'],
        'min_prob': kw['min_prob'],
        'holdout_only': kw['holdout_only'],
        'actual': kw['actual_only'],
        'bin_size': kw['bin_size'],
        'show': kw.get('show', True),
    })


@map_cmd.command('next')
@click.pass_context
@click.option('--run', 'run_id', default=None)
@click.option('-o', '--out-folder', type=click.Path(file_okay=False), default=None)
def map_next(ctx, run_id, out_folder):
    """Advance to the next time window."""
    _step_from_ctx(ctx, 1, out_folder, run_id)


@map_cmd.command('prev')
@click.pass_context
@click.option('--run', 'run_id', default=None)
@click.option('-o', '--out-folder', type=click.Path(file_okay=False), default=None)
def map_prev(ctx, run_id, out_folder):
    """Go back one time window."""
    _step_from_ctx(ctx, -1, out_folder, run_id)


@report.command('watch')
@click.option('--run', 'run_id', default=None, help='Run id [default: latest].')
@click.option(
    '-o', '--out-folder',
    type=click.Path(file_okay=False),
    default='fowd-reports',
    show_default=True,
)
@click.option('--bin', 'bin_size', default='1h', show_default=True)
@click.option('--delay', default=0.4, show_default=True, type=float,
              help='Seconds between frames.')
@click.option('--min-prob', default=0.0, show_default=True, type=float)
@click.option('--holdout-only', is_flag=True)
@click.option('--actual', is_flag=True)
@click.option('--show/--no-show', default=True, show_default=True)
def watch_cmd(run_id, out_folder, bin_size, delay, min_prob, holdout_only,
              actual, show):
    """Play the interactive world map, auto-advancing the time slider."""
    del holdout_only
    try:
        run_id, predictions = load_predictions(out_folder, run_id)
        png_path = _png_path(out_folder, run_id)
        interactive_world_map(
            predictions, run_id, png_path, show=show, bin_size=bin_size,
            min_prob=min_prob, actual_only=actual, autoplay=delay,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc))
    click.echo('Wrote {}'.format(png_path))
