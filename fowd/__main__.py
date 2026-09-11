"""Allow `python3 -m fowd` and `python3 fowd` when the CLI is not on PATH."""

from .cli import entrypoint

entrypoint()
