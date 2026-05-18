"""Read-only NumPy views over OEFP-owned native memory."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from typing import Any

import numpy as np


_CTYPES_BY_DTYPE: dict[np.dtype[Any], Any] = {
    np.dtype(np.uint64): ctypes.c_uint64,
    np.dtype(np.uint32): ctypes.c_uint32,
    np.dtype(np.int64): ctypes.c_int64,
    np.dtype(np.float64): ctypes.c_double,
}


def _shape_size(shape: Sequence[int]) -> int:
    size = 1
    for dimension in shape:
        size *= int(dimension)
    return size


def readonly_array_from_address(
    owner: Any,
    address: int,
    shape: Sequence[int],
    dtype: np.dtype,
) -> np.ndarray:
    """Create a read-only NumPy view and keep the native owner alive."""
    normalized_dtype = np.dtype(dtype)
    c_type = _CTYPES_BY_DTYPE[normalized_dtype]
    normalized_shape = tuple(int(dimension) for dimension in shape)
    size = _shape_size(normalized_shape)
    if size == 0:
        array = np.empty(normalized_shape, dtype=normalized_dtype)
        array.setflags(write=False)
        return array
    if address == 0:
        raise ValueError("Cannot create a non-empty array view from a null address.")

    buffer_type = c_type * size
    buffer = buffer_type.from_address(int(address))
    setattr(buffer, "_oefp_owner", owner)
    array = np.ctypeslib.as_array(buffer).reshape(normalized_shape)
    array.setflags(write=False)
    return array
