"""Runtime input boundary: never read test artifacts; own new checkpoints only."""
from contextlib import contextmanager
import builtins
import io
import os
from pathlib import Path
import re

@contextmanager
def input_guard(checkpoint_root=None):
    original=(builtins.open,io.open,os.open)
    owner=Path(checkpoint_root).resolve() if checkpoint_root is not None else None
    def check(name):
        if isinstance(name,int):return
        p=Path(os.fsdecode(name)).absolute()
        if any(re.search(r'(^|[-_])test([._-]|$)|(^|[-_])week_?26([._-]|$)',part.lower()) for candidate in (p,p.resolve()) for part in candidate.parts):raise ValueError('runtime test read blocked')
        if re.search(r'\.(pt|pth|ckpt)(\.|$)',p.name.lower()):
            if owner is None or not p.resolve().is_relative_to(owner) or any(x.is_symlink() for x in (p,*p.parents)):raise ValueError('checkpoint outside own new output blocked')
    def builtin(file,*a,**kw):check(file);return original[0](file,*a,**kw)
    def io_open(file,*a,**kw):check(file);return original[1](file,*a,**kw)
    def os_open(file,*a,**kw):check(file);return original[2](file,*a,**kw)
    builtins.open,io.open,os.open=builtin,io_open,os_open
    try:yield
    finally:builtins.open,io.open,os.open=original
