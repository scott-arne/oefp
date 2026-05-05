"""Public Python API for OEFP."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from . import oefp as _native
from ._views import readonly_array_from_address

_UINT32_MAX = 2**32 - 1


def _mode_value(mode: str) -> Any:
    normalized = mode.lower()
    if normalized == "similarity":
        return _native.MetricMode_Similarity
    if normalized == "distance":
        return _native.MetricMode_Distance
    raise ValueError(f"Unknown metric mode: {mode}")


def _batch_options(num_threads: int, chunk_size: int) -> Any:
    options = _native.BatchKernelOptions()
    options.num_threads = int(num_threads)
    options.chunk_size = int(chunk_size)
    return options


def _manual_spec(num_bits: int, algorithm: str) -> Any:
    spec = _native.FingerprintSpec()
    spec.size_bits = int(num_bits)
    spec.value_type = _native.FingerprintValueType_Binary
    spec.source_name = "OEFP"
    spec.source_type = algorithm
    spec.source_version = ""
    spec.parameters = ""
    return spec


def _uint32_option(name: str, value: Any, *, positive: bool) -> int:
    if not isinstance(value, Integral):
        raise TypeError(f"Morgan {name} must be an integer.")
    normalized = int(value)
    if positive:
        if normalized <= 0:
            raise ValueError(f"Morgan {name} must be greater than zero.")
    elif normalized < 0:
        raise ValueError(f"Morgan {name} must be non-negative.")
    if normalized > _UINT32_MAX:
        raise ValueError(f"Morgan {name} must be no greater than {_UINT32_MAX}.")
    return normalized


class OEFP:
    """Python wrapper for a native dense-binary OEFP."""

    def __init__(self, native: Any):
        self._native = native

    @classmethod
    def from_on_bits(
        cls,
        num_bits: int,
        bits: Sequence[int],
        *,
        algorithm: str = "manual",
    ) -> OEFP:
        """Create a dense binary fingerprint from on-bit indices."""
        native = _native._NativeOEFP(_manual_spec(num_bits, algorithm))
        for bit in bits:
            native.SetBit(int(bit))
        return cls(native)

    @property
    def words(self) -> np.ndarray:
        """Read-only view of the native uint64 word storage."""
        return readonly_array_from_address(
            self,
            self._native.WordDataAddress(),
            (self._native.WordCount(),),
            np.dtype(np.uint64),
        )

    @property
    def popcount(self) -> int:
        """Number of on bits."""
        return int(self._native.CountOnBits())

    @property
    def num_bits(self) -> int:
        """Fixed fingerprint size in bits."""
        return int(self._native.SizeBits())


class OEFPCount:
    """Python wrapper for a native sparse counted fingerprint."""

    def __init__(self, native: Any):
        self._native = native

    @property
    def indices(self) -> np.ndarray:
        """Read-only view of sorted nonzero count indices."""
        return readonly_array_from_address(
            self,
            self._native.IndexDataAddress(),
            (self._native.NonzeroCount(),),
            np.dtype(np.uint32),
        )

    @property
    def counts(self) -> np.ndarray:
        """Read-only view of counts parallel to indices."""
        return readonly_array_from_address(
            self,
            self._native.CountDataAddress(),
            (self._native.NonzeroCount(),),
            np.dtype(np.uint32),
        )

    @property
    def nonzero_count(self) -> int:
        """Number of nonzero sparse count entries."""
        return int(self._native.NonzeroCount())

    @property
    def total_count(self) -> int:
        """Sum of all sparse counts."""
        return int(self._native.TotalCount())

    @property
    def num_bits(self) -> int:
        """Fixed folded fingerprint size."""
        return int(self._native.SizeBits())


class OEFPBatch:
    """Python wrapper for a native dense-binary OEFPBatch."""

    def __init__(self, native: Any):
        self._native = native

    @classmethod
    def from_fingerprints(cls, fingerprints: Sequence[OEFP]) -> OEFPBatch:
        """Create a contiguous batch from dense binary fingerprints."""
        native_fps = _native.OEFPVector()
        for fp in fingerprints:
            native_fps.push_back(fp._native)
        return cls(_native._NativeOEFPBatch.FromFingerprints(native_fps))

    @property
    def words(self) -> np.ndarray:
        """Read-only view of the row-major uint64 word matrix."""
        return readonly_array_from_address(
            self,
            self._native.WordDataAddress(),
            (self._native.Size(), self._native.WordsPerFingerprint()),
            np.dtype(np.uint64),
        )

    @property
    def popcounts(self) -> np.ndarray:
        """Read-only view of one uint32 popcount per row."""
        return readonly_array_from_address(
            self,
            self._native.PopCountDataAddress(),
            (self._native.Size(),),
            np.dtype(np.uint32),
        )

    @property
    def size(self) -> int:
        """Number of fingerprint rows."""
        return int(self._native.Size())

    @property
    def num_bits(self) -> int:
        """Fixed fingerprint size in bits."""
        return int(self._native.SizeBits())


@dataclass(frozen=True)
class Metric:
    """Python wrapper for native OEFP metrics."""

    _native: Any

    @classmethod
    def tanimoto(cls, *, mode: str = "similarity") -> Metric:
        """Create a Tanimoto metric."""
        return cls(_native._NativeMetric.Tanimoto(_mode_value(mode)))

    @classmethod
    def jaccard(cls, *, mode: str = "similarity") -> Metric:
        """Create a Jaccard metric."""
        return cls(_native._NativeMetric.Jaccard(_mode_value(mode)))

    @classmethod
    def tversky(
        cls,
        alpha: float,
        beta: float,
        *,
        mode: str = "similarity",
    ) -> Metric:
        """Create a Tversky metric."""
        return cls(_native._NativeMetric.Tversky(alpha, beta, _mode_value(mode)))

    @classmethod
    def dice(cls, *, mode: str = "similarity") -> Metric:
        """Create a Dice metric."""
        return cls(_native._NativeMetric.Dice(_mode_value(mode)))

    @classmethod
    def cosine(cls, *, mode: str = "similarity") -> Metric:
        """Create a Cosine metric."""
        return cls(_native._NativeMetric.Cosine(_mode_value(mode)))

    @classmethod
    def manhattan(cls, *, mode: str = "distance") -> Metric:
        """Create a Manhattan metric."""
        return cls(_native._NativeMetric.Manhattan(_mode_value(mode)))


def compare(
    a: OEFP,
    b: OEFP | OEFPBatch,
    metric: Metric,
    *,
    num_threads: int = 0,
    chunk_size: int = 256,
) -> float | np.ndarray:
    """Compare one fingerprint with another fingerprint or batch."""
    if isinstance(b, OEFP):
        return float(_native.Compare(a._native, b._native, metric._native))

    output = np.empty((b.size,), dtype=np.float64)
    _native.CompareIntoAddress(
        a._native,
        b._native,
        metric._native,
        int(output.ctypes.data),
        output.size,
        _batch_options(num_threads, chunk_size),
    )
    return output


def cdist(
    a: OEFPBatch,
    b: OEFPBatch,
    metric: Metric,
    *,
    num_threads: int = 0,
    chunk_size: int = 256,
) -> np.ndarray:
    """Return row-major cross-distance/comparison values as a 2D array."""
    output = np.empty((a.size, b.size), dtype=np.float64)
    _native.CDistIntoAddress(
        a._native,
        b._native,
        metric._native,
        int(output.ctypes.data),
        output.size,
        _batch_options(num_threads, chunk_size),
    )
    return output


def pdist(
    batch: OEFPBatch,
    metric: Metric,
    *,
    num_threads: int = 0,
    chunk_size: int = 256,
) -> np.ndarray:
    """Return SciPy-compatible condensed pairwise values."""
    output = np.empty((batch.size * (batch.size - 1) // 2,), dtype=np.float64)
    _native.PDistIntoAddress(
        batch._native,
        metric._native,
        int(output.ctypes.data),
        output.size,
        _batch_options(num_threads, chunk_size),
    )
    return output


def _morgan_options(
    radius: int = 2,
    num_bits: int = 2048,
    use_chirality: bool = False,
    use_bond_types: bool = True,
    only_nonzero_invariants: bool = False,
    include_ring_membership: bool = True,
    include_redundant_environments: bool = False,
) -> Any:
    radius_int = _uint32_option("radius", radius, positive=False)
    num_bits_int = _uint32_option("num_bits", num_bits, positive=True)
    if use_chirality:
        raise ValueError("Morgan chirality conformance is not implemented yet.")
    options = _native.MorganOptions()
    options.radius = radius_int
    options.num_bits = num_bits_int
    options.use_chirality = bool(use_chirality)
    options.use_bond_types = bool(use_bond_types)
    options.only_nonzero_invariants = bool(only_nonzero_invariants)
    options.include_ring_membership = bool(include_ring_membership)
    options.include_redundant_environments = bool(include_redundant_environments)
    return options


def morgan_fingerprint(
    mol: Any,
    *,
    radius: int = 2,
    num_bits: int = 2048,
    use_chirality: bool = False,
    use_bond_types: bool = True,
    only_nonzero_invariants: bool = False,
    include_ring_membership: bool = True,
    include_redundant_environments: bool = False,
) -> OEFP:
    """Generate an RDKit-compatible folded binary Morgan fingerprint."""
    options = _morgan_options(
        radius,
        num_bits,
        use_chirality,
        use_bond_types,
        only_nonzero_invariants,
        include_ring_membership,
        include_redundant_environments,
    )
    return OEFP(_native.MakeMorganFingerprint(mol, options))


def morgan_count_fingerprint(
    mol: Any,
    *,
    radius: int = 2,
    num_bits: int = 2048,
    use_chirality: bool = False,
    use_bond_types: bool = True,
    only_nonzero_invariants: bool = False,
    include_ring_membership: bool = True,
    include_redundant_environments: bool = False,
) -> OEFPCount:
    """Generate an RDKit-compatible folded count Morgan fingerprint."""
    options = _morgan_options(
        radius,
        num_bits,
        use_chirality,
        use_bond_types,
        only_nonzero_invariants,
        include_ring_membership,
        include_redundant_environments,
    )
    return OEFPCount(_native.MakeMorganCountFingerprint(mol, options))


def from_openeye_fingerprint(fp: Any) -> OEFP:
    """Import an OpenEye OEFingerPrint as an OEFP."""
    return OEFP(_native.FromOEFingerPrint(fp))


def to_openeye_fingerprint(fp: OEFP) -> Any:
    """Export an OEFP as an OpenEye OEFingerPrint."""
    from openeye import oechem, oegraphsim

    spec = fp._native.Spec()
    if (
        spec.value_type != _native.FingerprintValueType_Binary
        or spec.source_name != "OpenEye"
        or not spec.source_type
    ):
        raise ValueError("OEFP spec does not contain OpenEye fingerprint type metadata.")

    fp_type = oegraphsim.OEGetFPType(spec.source_type)
    if fp_type is None:
        raise ValueError("OpenEye fingerprint type metadata could not be resolved.")

    # Resolve and assign the type through Python's OpenEye runtime so
    # OEIsSameFPType sees the same type identity as Python-created fingerprints.
    exported = oegraphsim.OEFingerPrint()
    byte_count = (fp.num_bits + 7) // 8
    data = oechem.OEUCharArray(fp.words.tobytes()[:byte_count])
    exported.SetData(data, fp.num_bits)
    exported.SetFPTypeBase(fp_type)
    return exported
