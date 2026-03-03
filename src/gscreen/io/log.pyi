import logging
from typing import IO, Any, ContextManager, Mapping, Optional

from ..typing import StrPath

__all__ = ["LoggerMixin", "redirect_all"]

class LoggerMixin:
    _logger: logging.Logger

    @classmethod
    def log(
        cls,
        lvl: int,
        msg: str,
        *args: Any,
        exc_info=...,
        stack_info: bool = ...,
        stacklevel: int = ...,
        extra: Optional[Mapping[str, object]] = ...,
    ) -> None: ...
    @classmethod
    def log_debug(
        cls,
        msg: str,
        exc_info=...,
        stack_info: bool = ...,
        stacklevel: int = ...,
        extra: Optional[Mapping[str, object]] = ...,
    ) -> None: ...
    @classmethod
    def log_info(
        cls,
        msg: str,
        *args: Any,
        exc_info=...,
        stack_info: bool = ...,
        stacklevel: int = ...,
        extra: Optional[Mapping[str, object]] = ...,
    ) -> None: ...
    @classmethod
    def log_warning(
        cls,
        msg: str,
        *args: Any,
        exc_info=...,
        stack_info: bool = ...,
        stacklevel: int = ...,
        extra: Optional[Mapping[str, object]] = ...,
    ) -> None: ...
    @classmethod
    def log_error(
        cls,
        msg: str,
        *args: Any,
        exc_info=...,
        stack_info: bool = ...,
        stacklevel: int = ...,
        extra: Optional[Mapping[str, object]] = ...,
    ) -> None: ...
    @classmethod
    def log_critical(
        cls,
        msg: str,
        *args: Any,
        exc_info=...,
        stack_info: bool = ...,
        stacklevel: int = ...,
        extra: Optional[Mapping[str, object]] = ...,
    ) -> None: ...

def redirect_file(src: IO, dst: IO) -> ContextManager[None]: ...
def redirect_all(
    dest: StrPath, mode: str = ..., **open_kwargs
) -> ContextManager[None]: ...
