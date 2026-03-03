import multiprocessing as mp
import os
import sys
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Optional, TypeVar, overload

import numpy as np

__all__ = [
    "get_nproc",
    "set_nproc",
    "nproc",
    "preconfigurable",
    "tolist_optional",
    "toarray_optional",
    "inside_debugger",
]

if "NSLOTS" in os.environ:
    _nproc = os.environ["NSLOTS"]
elif "SLURM_CPUS_PER_TASK" in os.environ:
    _nproc = int(os.environ["SLURM_CPUS_PER_TASK"])
    # If started with srun, use SLURM_CPUS_PER_TASK directly
    # else, use SLURM_CPUS_PER_TASK * SLURM_NTASKS (used for mpirun)
    if "SLURM_SRUN_COMM_HOST" not in os.environ:
        _nproc *= int(
            os.getenv("SLURM_NTASKS") or os.getenv("SLURM_NNODES") or 1
        )
elif "SLURM_NTASKS" in os.environ:
    _nproc = os.environ["SLURM_NTASKS"]
else:
    _nproc = 1

_PARAM_N_PROC = min(int(_nproc), mp.cpu_count())


def get_nproc() -> int:
    return _PARAM_N_PROC


def set_nproc(nproc: int):
    global _PARAM_N_PROC  # noqa: PLW0603
    _PARAM_N_PROC = nproc


@contextmanager
def nproc(n: int):
    old = get_nproc()
    oldenv = os.getenv("NSLOTS", None)
    try:
        set_nproc(n)
        os.environ["NSLOTS"] = str(n)
        yield
    finally:
        set_nproc(old)
        if oldenv is None:
            del os.environ["NSLOTS"]
        else:
            os.environ["NSLOTS"] = oldenv


_A = TypeVar("_A")
_R = TypeVar("_R")


def preconfigurable(func: Callable[[_A], _R], /):  # type: ignore
    @overload
    def outer_wrapper(**kwargs) -> Callable[[_A], _R]: ...

    @overload
    def outer_wrapper(arg: _A, /, **kwargs) -> _R: ...

    @wraps(func)
    def outer_wrapper(arg=None, /, **kwargs):
        # Hack from dataclasses.dataclass
        def inner_wrapper(arg: _A, /):
            return func(arg, **kwargs)

        if arg is None:
            # "pre-configuring" mode
            return inner_wrapper

        # "direct invocation" mode
        return inner_wrapper(arg)

    return outer_wrapper


def tolist_optional(a: Optional[np.ndarray]) -> Optional[list]:
    return None if a is None else a.tolist()


def toarray_optional(a: Optional[list]) -> Optional[np.ndarray]:
    return None if a is None else np.array(a)


if hasattr(sys, "gettrace"):

    def inside_debugger() -> bool:
        return sys.gettrace() is not None
else:

    def inside_debugger() -> bool:
        return False
