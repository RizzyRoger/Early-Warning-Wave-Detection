"""
progress.py

Command-line progress for discovery training and pipeline stages.
"""

import sys

NUM_BOOST_ROUNDS = 180


def _tqdm():
    import tqdm
    return tqdm


def stage_line(current, total, name):
    """Print a pipeline stage label to stderr."""
    sys.stderr.write('[{}/{}] {}\n'.format(current, total, name))
    sys.stderr.flush()


def training_bar(total=NUM_BOOST_ROUNDS, desc='Training', unit='round'):
    """Return a tqdm bar for boosting rounds."""
    return _tqdm().tqdm(
        total=int(total),
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        file=sys.stderr,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {unit}s [{elapsed}<{remaining}]',
    )


class CheckpointCallback(object):
    """LightGBM callback that writes the model every N rounds."""

    def __init__(self, directory, every, extension='txt'):
        self.directory = directory
        self.every = int(every) if every else 0
        self.extension = extension
        self.saved = []

    def __call__(self, env):
        if self.every < 1:
            return
        n = int(env.iteration) + 1
        if n % self.every != 0:
            return
        path = self._path(n)
        env.model.save_model(path)
        self.saved.append(path)

    def _path(self, n):
        import os
        return os.path.join(
            self.directory, 'round_{:06d}.{}'.format(n, self.extension)
        )


class BoostProgress(object):
    """LightGBM callback that advances a tqdm bar each boosting round."""

    def __init__(self, total=NUM_BOOST_ROUNDS, desc='Training'):
        self.pbar = training_bar(total=total, desc=desc)

    def __call__(self, env):
        extra = {}
        results = getattr(env, 'evaluation_result_list', None) or []
        for item in results:
            # (data_name, eval_name, result, is_higher_better)
            if len(item) >= 3:
                extra[item[1]] = '{:.4f}'.format(item[2])
        if extra:
            self.pbar.set_postfix(extra, refresh=False)
        self.pbar.update(1)

    def close(self):
        self.pbar.close()
