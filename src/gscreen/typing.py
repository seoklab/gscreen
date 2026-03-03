from os import PathLike
from typing import Iterator, Protocol, Tuple, TypeVar, Union

import numpy as np

StrPath = Union[str, PathLike[str]]

_T_co = TypeVar("_T_co", covariant=True)


class SizedIterable(Protocol[_T_co]):
    def __iter__(self) -> Iterator[_T_co]: ...
    def __len__(self) -> int: ...


Vector = np.ndarray[Tuple[int], np.dtype[np.float_]]
Points = np.ndarray[Tuple[int, int], np.dtype[np.float_]]
