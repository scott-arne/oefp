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


def _uint32_sequence(name: str, values: Sequence[Any]) -> list[int]:
    normalized_values: list[int] = []
    for value in values:
        if not isinstance(value, Integral):
            raise TypeError(f"Morgan {name} entries must be integers.")
        normalized = int(value)
        if normalized < 0:
            raise ValueError(f"Morgan {name} entries must be non-negative.")
        if normalized > _UINT32_MAX:
            raise ValueError(f"Morgan {name} entries must be no greater than {_UINT32_MAX}.")
        normalized_values.append(normalized)
    return normalized_values


def _native_uint32_vector(values: Sequence[int]) -> Any:
    vector = _native.UInt32Vector()
    for value in values:
        vector.push_back(int(value))
    return vector


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


class OEFPSparse:
    """Python wrapper for a native sparse binary fingerprint."""

    def __init__(self, native: Any):
        self._native = native

    @property
    def indices(self) -> np.ndarray:
        """Read-only view of sorted on-bit identifiers."""
        return readonly_array_from_address(
            self,
            self._native.IndexDataAddress(),
            (self._native.CountOnBits(),),
            np.dtype(np.uint32),
        )

    @property
    def popcount(self) -> int:
        """Number of on-bit entries."""
        return int(self._native.CountOnBits())

    @property
    def num_bits(self) -> int:
        """Sparse fingerprint identifier domain size."""
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


class OEFPCountBatch:
    """Python wrapper for a native sparse counted OEFPCountBatch."""

    def __init__(self, native: Any):
        self._native = native

    @classmethod
    def from_fingerprints(cls, fingerprints: Sequence[OEFPCount]) -> OEFPCountBatch:
        """Create a contiguous sparse batch from counted fingerprints."""
        native_fps = _native.OEFPCountVector()
        for fp in fingerprints:
            native_fps.push_back(fp._native)
        return cls(_native._NativeOEFPCountBatch.FromFingerprints(native_fps))

    @property
    def indices(self) -> np.ndarray:
        """Read-only view of flattened sparse uint32 indices."""
        return readonly_array_from_address(
            self,
            self._native.IndexDataAddress(),
            (self._native.EntryCount(),),
            np.dtype(np.uint32),
        )

    @property
    def counts(self) -> np.ndarray:
        """Read-only view of flattened sparse uint32 counts."""
        return readonly_array_from_address(
            self,
            self._native.CountDataAddress(),
            (self._native.EntryCount(),),
            np.dtype(np.uint32),
        )

    @property
    def offsets(self) -> np.ndarray:
        """Read-only view of row offsets into indices and counts."""
        return readonly_array_from_address(
            self,
            self._native.RowOffsetDataAddress(),
            (self._native.Size() + 1,),
            np.dtype(np.uint64),
        )

    @property
    def size(self) -> int:
        """Number of fingerprint rows."""
        return int(self._native.Size())

    @property
    def entry_count(self) -> int:
        """Number of flattened sparse count entries."""
        return int(self._native.EntryCount())

    @property
    def num_bits(self) -> int:
        """Fixed folded fingerprint size in bits."""
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
    a: OEFP | OEFPCount | OEFPSparse,
    b: OEFP | OEFPBatch | OEFPCount | OEFPCountBatch | OEFPSparse,
    metric: Metric,
    *,
    num_threads: int = 0,
    chunk_size: int = 256,
) -> float | np.ndarray:
    """Compare one fingerprint with another fingerprint or batch."""
    if isinstance(a, OEFP) and isinstance(b, OEFP):
        return float(_native.Compare(a._native, b._native, metric._native))
    if isinstance(a, OEFPCount) and isinstance(b, OEFPCount):
        return float(_native.Compare(a._native, b._native, metric._native))
    if isinstance(a, OEFPSparse) and isinstance(b, OEFPSparse):
        return float(_native.Compare(a._native, b._native, metric._native))
    if isinstance(a, OEFP) and isinstance(b, OEFPBatch):
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
    if isinstance(a, OEFPCount) and isinstance(b, OEFPCountBatch):
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

    raise TypeError(
        "compare expects OEFP/OEFP, OEFP/OEFPBatch, OEFPCount/OEFPCount, "
        "OEFPCount/OEFPCountBatch, or OEFPSparse/OEFPSparse inputs."
    )


def cdist(
    a: OEFPBatch | OEFPCountBatch,
    b: OEFPBatch | OEFPCountBatch,
    metric: Metric,
    *,
    num_threads: int = 0,
    chunk_size: int = 256,
) -> np.ndarray:
    """Return row-major cross-distance/comparison values as a 2D array."""
    if not (
        (isinstance(a, OEFPBatch) and isinstance(b, OEFPBatch))
        or (isinstance(a, OEFPCountBatch) and isinstance(b, OEFPCountBatch))
    ):
        raise TypeError(
            "cdist expects OEFPBatch/OEFPBatch or OEFPCountBatch/OEFPCountBatch inputs."
        )

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
    batch: OEFPBatch | OEFPCountBatch,
    metric: Metric,
    *,
    num_threads: int = 0,
    chunk_size: int = 256,
) -> np.ndarray:
    """Return SciPy-compatible condensed pairwise values."""
    if not isinstance(batch, (OEFPBatch, OEFPCountBatch)):
        raise TypeError("pdist expects an OEFPBatch or OEFPCountBatch input.")

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
    count_simulation: bool = False,
    count_bounds: Sequence[int] | None = None,
) -> Any:
    radius_int = _uint32_option("radius", radius, positive=False)
    num_bits_int = _uint32_option("num_bits", num_bits, positive=True)
    normalized_count_bounds = (
        [1, 2, 4, 8]
        if count_bounds is None
        else _uint32_sequence("count_bounds", count_bounds)
    )
    if count_simulation and not normalized_count_bounds:
        raise ValueError("Morgan count_bounds cannot be empty when count simulation is enabled.")
    if count_simulation and len(normalized_count_bounds) >= num_bits_int:
        raise ValueError("Morgan count_bounds length must be smaller than num_bits.")
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
    options.count_simulation = bool(count_simulation)
    options.count_bounds = _native_uint32_vector(normalized_count_bounds)
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
    count_simulation: bool = False,
    count_bounds: Sequence[int] | None = None,
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
        count_simulation,
        count_bounds,
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


def morgan_sparse_count_fingerprint(
    mol: Any,
    *,
    radius: int = 2,
    use_chirality: bool = False,
    use_bond_types: bool = True,
    only_nonzero_invariants: bool = False,
    include_ring_membership: bool = True,
    include_redundant_environments: bool = False,
) -> OEFPCount:
    """Generate an RDKit-compatible sparse count Morgan fingerprint."""
    options = _morgan_options(
        radius,
        2048,
        use_chirality,
        use_bond_types,
        only_nonzero_invariants,
        include_ring_membership,
        include_redundant_environments,
    )
    return OEFPCount(_native.MakeMorganSparseCountFingerprint(mol, options))


def morgan_sparse_fingerprint(
    mol: Any,
    *,
    radius: int = 2,
    use_chirality: bool = False,
    use_bond_types: bool = True,
    only_nonzero_invariants: bool = False,
    include_ring_membership: bool = True,
    include_redundant_environments: bool = False,
) -> OEFPSparse:
    """Generate an RDKit-compatible sparse binary Morgan fingerprint."""
    options = _morgan_options(
        radius,
        2048,
        use_chirality,
        use_bond_types,
        only_nonzero_invariants,
        include_ring_membership,
        include_redundant_environments,
    )
    return OEFPSparse(_native.MakeMorganSparseFingerprint(mol, options))


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
