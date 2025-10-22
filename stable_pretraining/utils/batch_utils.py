"""Utility functions for handling batch and outputs dictionaries in callbacks."""

from typing import Any, Dict, Optional, Union, Iterable
import torch
import collections.abc
import dataclasses
import copy
from loguru import logger as logging


def get_data_from_batch_or_outputs(
    key: Union[Iterable[str], str],
    batch: Dict[str, Any],
    outputs: Optional[Dict[str, Any]] = None,
    caller_name: str = "Callback",
) -> Optional[Any]:
    """Get data from either outputs or batch dictionary.

    In PyTorch Lightning, the outputs parameter in callbacks contains the return
    value from training_step/validation_step, while batch contains the original
    input. Since forward methods may modify batch in-place but Lightning creates
    a copy for outputs, we need to check both.

    Args:
        key: The key(s) to look for in the dictionaries
        batch: The original batch dictionary
        outputs: The outputs dictionary from training/validation step
        caller_name: Name of the calling function/class for logging

    Returns:
        The data associated with the key, or None if not found
    """
    output_as_list = True
    if type(key) is str:
        key = [key]
        output_as_list = False
    out = []
    for k in key:
        # First check outputs (which contains the forward pass results)
        if outputs is not None and k in outputs:
            out.append(outputs[k])
        elif k in batch:
            out.append(batch[k])
        else:
            msg = (
                f"{caller_name}: Key '{k}' not found in batch or outputs. "
                f"Available batch keys: {list(batch.keys())}, "
                f"Available output keys: {list(outputs.keys()) if outputs else 'None'}"
            )
            logging.warning(msg)
            raise ValueError(msg)
    if output_as_list:
        return out
    return out[0]


def detach_tensors(obj: Any) -> Any:
    """Recursively traverse an object and return an equivalent structure with all torch tensors detached.

    - Preserves structure, types, and shared references.
    - Handles cycles and arbitrary Python objects (including __dict__ and __slots__).
    - Does not mutate the input; only rebuilds containers if needed.
    - torch.nn.Parameter is replaced with a detached Tensor (not Parameter).
    - Optionally supports attrs classes if 'attr' is installed.

    Args:
        obj: The input object (can be arbitrarily nested).

    Returns:
        A new object with all torch tensors detached, or the original object if no tensors found.
    Performance notes:
        - Uses memoization to avoid redundant work and preserve shared/cyclic structure.
        - Avoids unnecessary copies: unchanged subtrees are returned as-is (same id).
        - Shallow-copies objects with __dict__ or __slots__ (does not call __init__).
    """
    memo: Dict[int, Any] = {}
    # Feature-detect attrs support
    try:
        import attr

        _HAS_ATTRS = True
    except ImportError:
        _HAS_ATTRS = False

    def _detach_impl(o: Any) -> Any:
        oid = id(o)
        if oid in memo:
            return memo[oid]
        # Tensors (including Parameter)
        if isinstance(o, torch.Tensor):
            result = o.detach()
            memo[oid] = result
            return result
        # defaultdict: must preserve default_factory and handle cycles
        if isinstance(o, collections.defaultdict):
            result = type(o)(o.default_factory)
            memo[oid] = result
            changed = False
            for k, v in o.items():
                new_v = _detach_impl(v)
                changed = changed or (new_v is not v)
                result[k] = new_v
            # Always return the new result, even if not changed, to ensure correct default_factory and keys
            return result
        # dict/OrderedDict/other Mapping (excluding defaultdict)
        if isinstance(o, collections.abc.Mapping):
            # For custom mapping subclasses, try to preserve type
            result = type(o)()
            memo[oid] = result
            changed = False
            for k, v in o.items():
                new_v = _detach_impl(v)
                changed = changed or (new_v is not v)
                result[k] = new_v
            # For plain dict, if nothing changed, return original
            if not changed and type(o) is dict:
                memo[oid] = o
                return o
            return result
        # Dataclasses (handle frozen and init=False fields)
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            # Step 1: create a shallow copy via dataclasses.replace (no field overrides)
            try:
                copy_obj = dataclasses.replace(o)
            except Exception:
                # fallback for dataclasses with no fields
                copy_obj = copy.copy(o)
            memo[oid] = copy_obj
            changed = False
            for f in dataclasses.fields(o):
                v = getattr(o, f.name)
                new_v = _detach_impl(v)
                if new_v is not v:
                    object.__setattr__(copy_obj, f.name, new_v)
                    changed = True
            if not changed:
                memo[oid] = o
                return o
            return copy_obj
        # attrs classes (if available)
        if _HAS_ATTRS and attr.has(o) and not isinstance(o, type):
            # Use attr.evolve to create a shallow copy, then set fields
            copy_obj = attr.evolve(o)
            memo[oid] = copy_obj
            changed = False
            for f in attr.fields(type(o)):
                v = getattr(o, f.name)
                new_v = _detach_impl(v)
                if new_v is not v:
                    object.__setattr__(copy_obj, f.name, new_v)
                    changed = True
            if not changed:
                memo[oid] = o
                return o
            return copy_obj
        # Namedtuple (but not plain tuple)
        if isinstance(o, tuple) and hasattr(o, "_fields"):
            values = []
            changed = False
            for v in o:
                new_v = _detach_impl(v)
                changed = changed or (new_v is not v)
                values.append(new_v)
            if not changed:
                memo[oid] = o
                return o
            result = type(o)(*values)
            memo[oid] = result
            return result
        # List
        if isinstance(o, list):
            result = []
            memo[oid] = result
            changed = False
            for v in o:
                new_v = _detach_impl(v)
                changed = changed or (new_v is not v)
                result.append(new_v)
            if not changed:
                memo[oid] = o
                return o
            return result
        # Tuple (not namedtuple)
        if isinstance(o, tuple):
            values = []
            changed = False
            for v in o:
                new_v = _detach_impl(v)
                changed = changed or (new_v is not v)
                values.append(new_v)
            if not changed:
                memo[oid] = o
                return o
            result = tuple(values)
            memo[oid] = result
            return result
        # Set
        if isinstance(o, set):
            result = set()
            memo[oid] = result
            changed = False
            for v in o:
                new_v = _detach_impl(v)
                changed = changed or (new_v is not v)
                result.add(new_v)
            if not changed:
                memo[oid] = o
                return o
            return result
        # Frozenset
        if isinstance(o, frozenset):
            values = []
            changed = False
            for v in o:
                new_v = _detach_impl(v)
                changed = changed or (new_v is not v)
                values.append(new_v)
            if not changed:
                memo[oid] = o
                return o
            result = frozenset(values)
            memo[oid] = result
            return result
        # Generic objects with __dict__ or __slots__
        if hasattr(o, "__dict__") or hasattr(o, "__slots__"):
            result = copy.copy(o)
            memo[oid] = result
            changed = False
            # __dict__ attributes
            if hasattr(result, "__dict__"):
                for k, v in result.__dict__.items():
                    new_v = _detach_impl(v)
                    if new_v is not v:
                        setattr(result, k, new_v)
                        changed = True
            # __slots__ attributes
            if hasattr(result, "__slots__"):
                for slot in result.__slots__:
                    if hasattr(result, slot):
                        v = getattr(result, slot)
                        new_v = _detach_impl(v)
                        if new_v is not v:
                            setattr(result, slot, new_v)
                            changed = True
            if not changed:
                memo[oid] = o
                return o
            return result
        # All other types: return as is
        memo[oid] = o
        return o

    return _detach_impl(obj)
