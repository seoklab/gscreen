from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type, Union

__all__ = ["SerializableObj", "SerializableDic", "Serializable"]

SerializablePrm = Union[str, int, bool, float, None]
SerializableVal = Union[SerializablePrm, "SerializableObj"]
SerializableDic = Dict[Union[str, Type["Serializable"]], SerializableVal]
SerializableSeq = Union[List[SerializableVal], Tuple[SerializableVal, ...]]
SerializableObj = Union[SerializableDic, SerializableSeq, "Serializable"]

_registry: Dict[str, Type["Serializable"]] = {}


class Serializable(ABC):
    @abstractmethod
    def __getstate__(self) -> SerializableObj:
        pass

    @abstractmethod
    def __setstate__(self, state: SerializableObj) -> None:
        pass

    @classmethod
    @property
    def type(cls):
        return f"gscreen_{cls.__name__.lower()}"

    def __init_subclass__(cls, /, register: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)

        if register:
            regname = cls.type
            if regname in _registry:
                raise ValueError(f"Duplicate registration name: {regname}")

            _registry[regname] = cls
