"""
__init__.py

Initialize global setup
"""

# we use multiprocessing, so prevent other libraries from using threads
import os
os.environ['OMP_NUM_THREADS'] = '1'
del os

# get version
try:
    from fowd._version import version as __version__  # noqa: F401
except ImportError:
    __version__ = '0.5.2'
