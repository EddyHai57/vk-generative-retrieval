"""R1 immutable code and historical accepted input channel have separate roots.

The existing G3 loader and every historical data pin remain unchanged. The caller
uses this directory only for historical input validation; model/source binding
always hashes the new code checkout independently.
"""
import os
from pathlib import Path


def data_root(source_root):
    value = os.environ.get('VKGR_DATA_ROOT')
    if value is not None and not value.strip():
        raise ValueError('VKGR_DATA_ROOT must name an existing accepted data channel')
    try:
        path = Path(source_root if value is None else value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError('VKGR_DATA_ROOT must name an existing accepted data channel') from exc
    if not path.is_dir():
        raise ValueError('VKGR_DATA_ROOT must name an existing accepted data channel')
    return path
