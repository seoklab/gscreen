import json
import logging
from typing import Type, Union

from .serialize import (
    Serializable,
    SerializableDic,
    SerializableObj,
    _registry,
)

__all__ = ["load", "loads", "dump", "dumps"]

_version = "1.0"
_logger = logging.getLogger(__name__)


class Serializer(json.JSONEncoder):
    @staticmethod
    def key_default(k):
        try:
            if issubclass(k, Serializable):
                return k.type
        except TypeError:
            pass
        return k

    def key_encode(self, obj: SerializableObj) -> SerializableObj:
        if isinstance(obj, dict):
            return {
                self.key_default(k): self.key_encode(v) for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [self.key_encode(o) for o in obj]
        return obj

    def default(self, obj):
        if isinstance(obj, Serializable):
            _logger.debug("Serializing %s", obj)
            return self.key_encode(obj.__getstate__())

        return super().default(obj)

    def encode(self, obj: SerializableDic) -> str:
        obj = obj.copy()
        obj["gscreen"] = _version
        return super().encode(self.key_encode(obj))


def _decode_object(cls: Type[Serializable], state: SerializableObj):
    state = _decode_dispatch(state)
    _logger.debug("In _decode_object(): %s <- %s", cls.type, state)
    obj = object.__new__(cls)
    try:
        obj.__setstate__(state)
    except (KeyError, TypeError):
        obj = state
    return obj


def _decode_dict(d: SerializableDic) -> SerializableObj:
    _logger.debug("In _decode_dict(): %s", d)
    ret = {}

    for k, v in d.items():
        _logger.debug(" In _decode_dict(): %s -> %s", k, v)
        try:
            cls = _registry[k]
        except KeyError:
            ret[k] = _decode_dispatch(v)
            continue

        if isinstance(v, list):
            ret[cls] = [_decode_object(cls, item) for item in v]
        else:
            ret[cls] = _decode_object(cls, v)

    return ret


def _decode_dispatch(o: SerializableObj):
    _logger.debug("In _decode_dispatch(): %s", o)
    if isinstance(o, dict):
        return _decode_dict(o)
    if isinstance(o, list):
        return [_decode_dispatch(v) for v in o]
    return o


def obj_hook(raw: dict) -> Union[dict, SerializableObj]:
    try:
        obj_ver = raw["gscreen"]
    except Exception:
        return raw

    if obj_ver != _version:
        raise ValueError(f"Invalid gscreen version: {obj_ver}")

    raw.pop("gscreen")
    # _logger.debug("Decoding object: %s", raw)
    return _decode_dict(raw)


def load(fp, object_hook=obj_hook, **kwargs) -> SerializableDic:
    return json.load(fp, object_hook=object_hook, **kwargs)


def loads(s, object_hook=obj_hook, **kwargs) -> SerializableDic:
    return json.loads(s, object_hook=object_hook, **kwargs)


def dump(obj: SerializableDic, fp, cls=Serializer, **kwargs):
    return fp.write(dumps(obj, cls, **kwargs))


def dumps(obj: SerializableDic, cls=Serializer, **kwargs):
    return cls(**kwargs).encode(obj)
