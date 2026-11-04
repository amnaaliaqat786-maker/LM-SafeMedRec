"""Restricted loader for trusted GAMENet pickle files.

The original GAMENet vocabulary file may contain a Voc object serialized
using dill. This module uses a custom Unpickler that permits selected
NumPy reconstruction helpers and replaces unsupported classes with
inert placeholder objects.

The loader then extracts the vocabulary mappings required by
LM-SafeMedRec.

Security note:
This is a restricted compatibility loader, not a guarantee that
arbitrary or untrusted pickle files are safe to deserialize. Only use
pickle files obtained from trusted sources.
"""

import pickle
import numpy as np

_ALLOWED_NUMPY = {
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
}


class StubObj:
    """Inert placeholder object: stores whatever state the pickle gives it."""
    _args = ()

    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state


def _stub_factory(name):
    def factory(*args, **kwargs):
        # Return a fresh, empty class so NEWOBJ/REDUCE can use it as a type.
        return type("Stub_" + name.split(".")[-1], (StubObj,), {"_args": args})
    return factory


Stub = StubObj


class _NpModule:
    def __init__(self, name):
        self.name = name


def _import_module(name, *_, **__):
    if not name.startswith("numpy"):
        raise pickle.UnpicklingError(f"blocked module import: {name}")
    return _NpModule(name)


def _get_attr(obj, attr):
    if isinstance(obj, _NpModule) and attr == "_reconstruct":
        from numpy.core.multiarray import _reconstruct  # noqa
        return _reconstruct
    if isinstance(obj, _NpModule) and attr in ("dtype", "ndarray"):
        return getattr(np, attr)
    raise pickle.UnpicklingError(f"blocked attribute: {attr}")


def _create_array(f, args, state, npdict=None):
    arr = f(*args)
    arr.__setstate__(state)
    return arr


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED_NUMPY:
            import importlib
            mod = importlib.import_module(module.replace("numpy.core", "numpy.core"))
            return getattr(mod, name)
        if module == "dill._dill" and name == "_import_module":
            return _import_module
        if module == "dill._dill" and name == "_get_attr":
            return _get_attr
        if module == "dill._dill" and name == "_create_array":
            return _create_array
        # anything else (functions, classes, types) becomes inert data
        return _stub_factory(f"{module}.{name}")


def load(path):
    with open(path, "rb") as fh:
        return SafeUnpickler(fh).load()


def find_vocabs(obj, found=None, depth=0):
    """Walk stubs and dicts and pull out every idx2word mapping."""
    if found is None:
        found = {}
    if depth > 12:
        return found
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "idx2word" and isinstance(v, dict):
                found.setdefault("_pending", []).append(v)
            find_vocabs(v, found, depth + 1)
            if isinstance(k, str) and k.endswith("_voc"):
                found.setdefault("_order", []).append(k)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            find_vocabs(v, found, depth + 1)
    elif isinstance(obj, StubObj):
        find_vocabs(getattr(obj, "state", None), found, depth + 1)
        find_vocabs(getattr(obj, "__dict__", None), found, depth + 1)
    return found


def load_vocab(path):
    """Return {'diag_voc': [codes...], 'pro_voc': [...], 'med_voc': [...]}."""
    raw = load(path)
    out = {}
    if isinstance(raw, dict):
        for key, val in raw.items():
            f = find_vocabs(val)
            maps = f.get("_pending", [])
            if maps:
                m = maps[0]
                out[key] = [m[i] for i in range(len(m))]
    return out
