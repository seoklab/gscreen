import logging
import os
import sys
from contextlib import contextmanager
from typing import IO

from ..typing import StrPath

__all__ = ["LoggerMixin", "redirect_all"]


class LoggerMixin:
    _logger: logging.Logger

    def __init_subclass__(cls):
        super().__init_subclass__()

        module = getattr(cls, "__module__", None)
        if module == "__main__":
            if sys.argv[0] == "-c":
                module = "stdin"
            else:
                module = os.path.basename(sys.argv[0])

        cls._logger = logging.getLogger(module)

    @classmethod
    def log(cls, lvl: int, msg: str, *args, **kwargs):
        cls._logger.log(lvl, msg, *args, **kwargs)

    @classmethod
    def log_debug(cls, msg: str, *args, **kwargs):
        cls.log(logging.DEBUG, msg, *args, **kwargs)

    @classmethod
    def log_info(cls, msg: str, *args, **kwargs):
        cls.log(logging.INFO, msg, *args, **kwargs)

    @classmethod
    def log_warning(cls, msg: str, *args, **kwargs):
        cls.log(logging.WARNING, msg, *args, **kwargs)

    @classmethod
    def log_error(cls, msg: str, *args, **kwargs):
        cls.log(logging.ERROR, msg, *args, **kwargs)

    @classmethod
    def log_critical(cls, msg: str, *args, **kwargs):
        cls.log(logging.CRITICAL, msg, *args, **kwargs)


@contextmanager
def redirect_file(src: IO, dst: IO):
    src_fd = src.fileno()
    with os.fdopen(os.dup(src_fd), "wb") as backup:
        src.flush()
        dst_fd = dst.fileno()
        try:
            os.dup2(dst_fd, src_fd)
            yield
        finally:
            src.flush()
            os.dup2(backup.fileno(), src_fd)


@contextmanager
def redirect_all(dest: StrPath, mode: str = "ab", **open_kwargs):
    """Redirect all stdout and stderr to a file."""
    with open(dest, mode=mode, **open_kwargs) as new_out:
        with (
            redirect_file(sys.stdout, new_out),
            redirect_file(sys.stderr, new_out),
        ):
            yield
