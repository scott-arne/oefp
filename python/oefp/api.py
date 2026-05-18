"""Public Python API for OEFP."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from typing import Any

import numpy as np

from . import _native
from ._views import readonly_array_from_address

_UINT32_MAX = 2**32 - 1
_NATIVE_TOKEN = object()


@dataclass(frozen=True)
class FingerprintSpec:
    """Read-only fingerprint metadata exposed by Python wrappers."""

    num_bits: int
    value_type: str
    source_name: str
    source_type: str
    source_version: str
    parameters: str
    source_type_id: int | None


@dataclass(frozen=True)
class DescriptorSpec:
    """Read-only descriptor metadata exposed by Python wrappers."""

    value_type: str
    source_name: str
    source_type: str
    source_version: str
    parameters: str


def _value_type_name(value: Any) -> str:
    if value == _native.FingerprintValueType_Binary:
        return "binary"
    if value == _native.FingerprintValueType_Counted:
        return "counted"
    return "unknown"


def _fingerprint_spec(native_spec: Any) -> FingerprintSpec:
    return FingerprintSpec(
        num_bits=int(native_spec.size_bits),
        value_type=_value_type_name(native_spec.value_type),
        source_name=str(native_spec.source_name),
        source_type=str(native_spec.source_type),
        source_version=str(native_spec.source_version),
        parameters=str(native_spec.parameters),
        source_type_id=int(native_spec.source_type_id)
        if native_spec.has_source_type_id
        else None,
    )


def _descriptor_value_type_name(value: Any) -> str:
    if value == _native.DescriptorValueType_Integer:
        return "integer"
    if value == _native.DescriptorValueType_Float:
        return "float"
    if value == _native.DescriptorValueType_String:
        return "string"
    return "unknown"


def _native_descriptor_value_type(value_type: str) -> Any:
    if value_type == "integer":
        return _native.DescriptorValueType_Integer
    if value_type == "float":
        return _native.DescriptorValueType_Float
    if value_type == "string":
        return _native.DescriptorValueType_String
    raise ValueError(f"Unknown descriptor value type: {value_type!r}.")


def _descriptor_spec(native_spec: Any) -> DescriptorSpec:
    return DescriptorSpec(
        value_type=_descriptor_value_type_name(native_spec.value_type),
        source_name=str(native_spec.source_name),
        source_type=str(native_spec.source_type),
        source_version=str(native_spec.source_version),
        parameters=str(native_spec.parameters),
    )


def _native_descriptor_spec(spec: DescriptorSpec) -> Any:
    native_spec = _native.DescriptorSpec()
    native_spec.value_type = _native_descriptor_value_type(spec.value_type)
    native_spec.source_name = spec.source_name
    native_spec.source_type = spec.source_type
    native_spec.source_version = spec.source_version
    native_spec.parameters = spec.parameters
    return native_spec


def _manual_descriptor_spec(
    value_type: str,
    source_name: str,
    source_type: str,
    source_version: str,
    parameters: str,
) -> DescriptorSpec:
    return DescriptorSpec(
        value_type=value_type,
        source_name=source_name,
        source_type=source_type,
        source_version=source_version,
        parameters=parameters,
    )


def _descriptor_mode_value(mode: str) -> Any:
    if mode == "count_overlap":
        return _native.DescriptorComparisonMode_CountOverlap
    if mode == "exact_count":
        return _native.DescriptorComparisonMode_ExactCount
    if mode == "presence":
        return _native.DescriptorComparisonMode_Presence
    raise ValueError(f"Unknown descriptor comparison mode: {mode!r}.")


def _metric_name_name(value: Any) -> str:
    names = {
        _native.MetricName_Euclidean: "euclidean",
        _native.MetricName_Manhattan: "manhattan",
        _native.MetricName_Chebyshev: "chebyshev",
        _native.MetricName_Minkowski: "minkowski",
        _native.MetricName_StandardizedEuclidean: "standardized_euclidean",
        _native.MetricName_Mahalanobis: "mahalanobis",
        _native.MetricName_Haversine: "haversine",
        _native.MetricName_Hamming: "hamming",
        _native.MetricName_Canberra: "canberra",
        _native.MetricName_BrayCurtis: "bray_curtis",
        _native.MetricName_Jaccard: "jaccard",
        _native.MetricName_Matching: "matching",
        _native.MetricName_Dice: "dice",
        _native.MetricName_Kulsinski: "kulsinski",
        _native.MetricName_RogersTanimoto: "rogers_tanimoto",
        _native.MetricName_RussellRao: "russell_rao",
        _native.MetricName_SokalMichener: "sokal_michener",
        _native.MetricName_SokalSneath: "sokal_sneath",
        _native.MetricName_Tanimoto: "tanimoto",
        _native.MetricName_Tversky: "tversky",
    }
    return names.get(value, "unknown")


def _metric_type_name(value: Any) -> str:
    if value == _native.MetricType_Distance:
        return "distance"
    if value == _native.MetricType_Similarity:
        return "similarity"
    return "unknown"


def _metric_space_name(value: Any) -> str:
    if value == _native.MetricSpace_Real:
        return "real"
    if value == _native.MetricSpace_Integer:
        return "integer"
    if value == _native.MetricSpace_Boolean:
        return "boolean"
    return "unknown"


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


def _uint32_option(context: str, name: str, value: Any, *, positive: bool) -> int:
    if not isinstance(value, Integral):
        raise TypeError(f"{context} {name} must be an integer.")
    normalized = int(value)
    if positive:
        if normalized <= 0:
            raise ValueError(f"{context} {name} must be greater than zero.")
    elif normalized < 0:
        raise ValueError(f"{context} {name} must be non-negative.")
    if normalized > _UINT32_MAX:
        raise ValueError(f"{context} {name} must be no greater than {_UINT32_MAX}.")
    return normalized


def _uint32_sequence(context: str, name: str, values: Sequence[Any]) -> list[int]:
    normalized_values: list[int] = []
    for value in values:
        if not isinstance(value, Integral):
            raise TypeError(f"{context} {name} entries must be integers.")
        normalized = int(value)
        if normalized < 0:
            raise ValueError(f"{context} {name} entries must be non-negative.")
        if normalized > _UINT32_MAX:
            raise ValueError(f"{context} {name} entries must be no greater than {_UINT32_MAX}.")
        normalized_values.append(normalized)
    return normalized_values


def _normalized_count_bounds(context: str, count_bounds: Sequence[int] | None) -> tuple[int, ...]:
    if count_bounds is None:
        return (1, 2, 4, 8)
    return tuple(_uint32_sequence(context, "count_bounds", count_bounds))


def _native_uint32_vector(values: Sequence[int]) -> Any:
    vector = _native.UInt32Vector()
    for value in values:
        vector.push_back(int(value))
    return vector


def _native_int64_vector(values: Sequence[int]) -> Any:
    vector = _native.Int64Vector()
    for value in values:
        vector.push_back(int(value))
    return vector


def _native_string_vector(values: Sequence[str]) -> Any:
    vector = _native.StringVector()
    for value in values:
        vector.push_back(str(value))
    return vector


def _native_double_vector(values: Sequence[float]) -> Any:
    vector = _native.DoubleVector()
    for value in values:
        vector.push_back(float(value))
    return vector


def _normalized_atom_pair_values(
    min_distance: int,
    max_distance: int,
    num_bits: int,
    use_chirality: bool,
    use_2d: bool,
    count_simulation: bool,
    count_bounds: Sequence[int] | None,
) -> tuple[int, int, int, bool, bool, bool, tuple[int, ...]]:
    min_distance_int = _uint32_option("Atom Pair", "min_distance", min_distance, positive=False)
    max_distance_int = _uint32_option("Atom Pair", "max_distance", max_distance, positive=False)
    num_bits_int = _uint32_option("Atom Pair", "num_bits", num_bits, positive=True)
    normalized_count_bounds = _normalized_count_bounds("Atom Pair", count_bounds)
    if min_distance_int > max_distance_int:
        raise ValueError("Atom Pair min_distance cannot exceed max_distance.")
    if max_distance_int >= 31:
        raise ValueError("Atom Pair max_distance must be smaller than 31.")
    if use_chirality:
        raise ValueError("Atom Pair chirality conformance is not implemented yet.")
    if not use_2d:
        raise ValueError("Atom Pair 3D distance conformance is not implemented yet.")
    if count_simulation and not normalized_count_bounds:
        raise ValueError("Atom Pair count_bounds cannot be empty when count simulation is enabled.")
    if count_simulation and len(normalized_count_bounds) >= num_bits_int:
        raise ValueError("Atom Pair count_bounds length must be smaller than num_bits.")

    return (
        min_distance_int,
        max_distance_int,
        num_bits_int,
        bool(use_chirality),
        bool(use_2d),
        bool(count_simulation),
        normalized_count_bounds,
    )


def _atom_pair_options(
    min_distance: int,
    max_distance: int,
    num_bits: int,
    use_chirality: bool,
    use_2d: bool,
    count_simulation: bool,
    count_bounds: Sequence[int] | None,
) -> Any:
    (
        min_distance_int,
        max_distance_int,
        num_bits_int,
        use_chirality_bool,
        use_2d_bool,
        count_simulation_bool,
        normalized_count_bounds,
    ) = _normalized_atom_pair_values(
        min_distance,
        max_distance,
        num_bits,
        use_chirality,
        use_2d,
        count_simulation,
        count_bounds,
    )
    options = _native.AtomPairOptions()
    options.min_distance = min_distance_int
    options.max_distance = max_distance_int
    options.num_bits = num_bits_int
    options.use_chirality = use_chirality_bool
    options.use_2d = use_2d_bool
    options.count_simulation = count_simulation_bool
    options.count_bounds = _native_uint32_vector(normalized_count_bounds)
    return options


class OEFP:
    """Python wrapper for a native dense-binary OEFP."""

    def __init__(self, native: Any, *, _token: object | None = None):
        if _token is not _NATIVE_TOKEN:
            raise TypeError(
                "OEFP objects are created by OEFP.from_on_bits(), "
                "morgan_fingerprint(), atom_pair_fingerprint(), or "
                "from_openeye_fingerprint()."
            )
        self._native = native

    @classmethod
    def _from_native(cls, native: Any) -> OEFP:
        return cls(native, _token=_NATIVE_TOKEN)

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
        return cls._from_native(native)

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

    @property
    def spec(self) -> FingerprintSpec:
        """Read-only fingerprint metadata."""
        return _fingerprint_spec(self._native.Spec())


class OEFPCount:
    """Python wrapper for a native sparse counted fingerprint."""

    def __init__(self, native: Any, *, _token: object | None = None):
        if _token is not _NATIVE_TOKEN:
            raise TypeError(
                "OEFPCount objects are created by OEFPCount factories such as "
                "morgan_count_fingerprint() or atom_pair_count_fingerprint()."
            )
        self._native = native

    @classmethod
    def _from_native(cls, native: Any) -> OEFPCount:
        return cls(native, _token=_NATIVE_TOKEN)

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

    @property
    def spec(self) -> FingerprintSpec:
        """Read-only fingerprint metadata."""
        return _fingerprint_spec(self._native.Spec())


class OEFPSparse:
    """Python wrapper for a native sparse binary fingerprint."""

    def __init__(self, native: Any, *, _token: object | None = None):
        if _token is not _NATIVE_TOKEN:
            raise TypeError(
                "OEFPSparse objects are created by OEFPSparse factories such as "
                "morgan_sparse_fingerprint() or atom_pair_sparse_fingerprint()."
            )
        self._native = native

    @classmethod
    def _from_native(cls, native: Any) -> OEFPSparse:
        return cls(native, _token=_NATIVE_TOKEN)

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

    @property
    def spec(self) -> FingerprintSpec:
        """Read-only fingerprint metadata."""
        return _fingerprint_spec(self._native.Spec())


class DescriptorSet:
    """Python wrapper for a native descriptor set."""

    def __init__(self, native: Any, *, _token: object | None = None):
        if _token is not _NATIVE_TOKEN:
            raise TypeError(
                "DescriptorSet objects are created by DescriptorSet.from_strings(), "
                "DescriptorSet.from_integers(), DescriptorSet.from_floats(), or "
                "descriptor factories."
            )
        self._native = native

    @classmethod
    def _from_native(cls, native: Any) -> DescriptorSet:
        return cls(native, _token=_NATIVE_TOKEN)

    @classmethod
    def from_strings(
        cls,
        keys: Sequence[str],
        counts: Sequence[int] | None = None,
        *,
        spec: DescriptorSpec | None = None,
        source_name: str = "OEFP",
        source_type: str = "manual",
        source_version: str = "",
        parameters: str = "",
    ) -> DescriptorSet:
        """Create counted string-key descriptors."""
        descriptor_spec = spec or _manual_descriptor_spec(
            "string",
            source_name,
            source_type,
            source_version,
            parameters,
        )
        native_spec = _native_descriptor_spec(descriptor_spec)
        native_keys = _native_string_vector(keys)
        if counts is None:
            return cls._from_native(_native._NativeDescriptorSet.FromStrings(native_spec, native_keys))
        return cls._from_native(
            _native._NativeDescriptorSet.FromStringCounts(
                native_spec,
                native_keys,
                _native_uint32_vector(counts),
            )
        )

    @classmethod
    def from_integers(
        cls,
        keys: Sequence[int],
        counts: Sequence[int] | None = None,
        *,
        spec: DescriptorSpec | None = None,
        source_name: str = "OEFP",
        source_type: str = "manual",
        source_version: str = "",
        parameters: str = "",
    ) -> DescriptorSet:
        """Create counted integer-key descriptors."""
        descriptor_spec = spec or _manual_descriptor_spec(
            "integer",
            source_name,
            source_type,
            source_version,
            parameters,
        )
        native_spec = _native_descriptor_spec(descriptor_spec)
        native_keys = _native_int64_vector(keys)
        if counts is None:
            return cls._from_native(_native._NativeDescriptorSet.FromIntegers(native_spec, native_keys))
        return cls._from_native(
            _native._NativeDescriptorSet.FromIntegerCounts(
                native_spec,
                native_keys,
                _native_uint32_vector(counts),
            )
        )

    @classmethod
    def from_floats(
        cls,
        keys: Sequence[float],
        counts: Sequence[int] | None = None,
        *,
        spec: DescriptorSpec | None = None,
        source_name: str = "OEFP",
        source_type: str = "manual",
        source_version: str = "",
        parameters: str = "",
    ) -> DescriptorSet:
        """Create counted float-key descriptors."""
        descriptor_spec = spec or _manual_descriptor_spec(
            "float",
            source_name,
            source_type,
            source_version,
            parameters,
        )
        native_spec = _native_descriptor_spec(descriptor_spec)
        native_keys = _native_double_vector(keys)
        if counts is None:
            return cls._from_native(_native._NativeDescriptorSet.FromFloats(native_spec, native_keys))
        return cls._from_native(
            _native._NativeDescriptorSet.FromFloatCounts(
                native_spec,
                native_keys,
                _native_uint32_vector(counts),
            )
        )

    @property
    def value_type(self) -> str:
        """Descriptor key value type."""
        return _descriptor_value_type_name(self._native.ValueType())

    @property
    def size(self) -> int:
        """Number of unique descriptor keys."""
        return int(self._native.Size())

    @property
    def total_count(self) -> int:
        """Sum of all descriptor counts."""
        return int(self._native.TotalCount())

    @property
    def string_keys(self) -> tuple[str, ...]:
        """Canonical sorted string keys."""
        if self.value_type != "string":
            return ()
        return tuple(str(value) for value in self._native.StringKeys())

    @property
    def integer_keys(self) -> tuple[int, ...]:
        """Canonical sorted integer keys."""
        if self.value_type != "integer":
            return ()
        keys = readonly_array_from_address(
            self,
            self._native.IntegerKeyDataAddress(),
            (self._native.Size(),),
            np.dtype(np.int64),
        )
        return tuple(int(value) for value in keys)

    @property
    def float_keys(self) -> tuple[float, ...]:
        """Canonical sorted float keys."""
        if self.value_type != "float":
            return ()
        keys = readonly_array_from_address(
            self,
            self._native.FloatKeyDataAddress(),
            (self._native.Size(),),
            np.dtype(np.float64),
        )
        return tuple(float(value) for value in keys)

    @property
    def counts(self) -> np.ndarray:
        """Read-only view of descriptor counts parallel to active keys."""
        return readonly_array_from_address(
            self,
            self._native.CountDataAddress(),
            (self._native.Size(),),
            np.dtype(np.uint32),
        )

    @property
    def spec(self) -> DescriptorSpec:
        """Read-only descriptor metadata."""
        return _descriptor_spec(self._native.Spec())


class DescriptorBatch:
    """Python wrapper for a native descriptor batch."""

    def __init__(self, native: Any, *, _token: object | None = None):
        if _token is not _NATIVE_TOKEN:
            raise TypeError(
                "DescriptorBatch objects are created by "
                "DescriptorBatch.from_descriptors()."
            )
        self._native = native

    @classmethod
    def _from_native(cls, native: Any) -> DescriptorBatch:
        return cls(native, _token=_NATIVE_TOKEN)

    @classmethod
    def from_descriptors(cls, descriptors: Sequence[DescriptorSet]) -> DescriptorBatch:
        """Create a contiguous descriptor batch from descriptor sets."""
        native_descriptors = _native.DescriptorSetVector()
        for descriptor_set in descriptors:
            native_descriptors.push_back(descriptor_set._native)
        return cls._from_native(_native._NativeDescriptorBatch.FromDescriptorSets(native_descriptors))

    @property
    def value_type(self) -> str:
        """Shared descriptor key value type."""
        return _descriptor_value_type_name(self._native.ValueType())

    @property
    def size(self) -> int:
        """Number of descriptor rows."""
        return int(self._native.Size())

    @property
    def entry_count(self) -> int:
        """Number of flattened descriptor entries."""
        return int(self._native.EntryCount())

    @property
    def string_keys(self) -> tuple[str, ...]:
        """Flattened string keys."""
        if self.value_type != "string":
            return ()
        return tuple(str(value) for value in self._native.StringKeys())

    @property
    def integer_keys(self) -> tuple[int, ...]:
        """Flattened integer keys."""
        if self.value_type != "integer":
            return ()
        keys = readonly_array_from_address(
            self,
            self._native.IntegerKeyDataAddress(),
            (self._native.EntryCount(),),
            np.dtype(np.int64),
        )
        return tuple(int(value) for value in keys)

    @property
    def float_keys(self) -> tuple[float, ...]:
        """Flattened float keys."""
        if self.value_type != "float":
            return ()
        keys = readonly_array_from_address(
            self,
            self._native.FloatKeyDataAddress(),
            (self._native.EntryCount(),),
            np.dtype(np.float64),
        )
        return tuple(float(value) for value in keys)

    @property
    def counts(self) -> np.ndarray:
        """Read-only view of flattened descriptor counts."""
        return readonly_array_from_address(
            self,
            self._native.CountDataAddress(),
            (self._native.EntryCount(),),
            np.dtype(np.uint32),
        )

    @property
    def offsets(self) -> np.ndarray:
        """Read-only view of row offsets into flattened keys and counts."""
        return readonly_array_from_address(
            self,
            self._native.RowOffsetDataAddress(),
            (self._native.Size() + 1,),
            np.dtype(np.uint64),
        )

    @property
    def spec(self) -> DescriptorSpec:
        """Read-only descriptor metadata for all rows."""
        return _descriptor_spec(self._native.Spec())


class OEFPMappingSet:
    """Python wrapper for native fingerprint bit environment mappings."""

    def __init__(self, native: Any, *, _token: object | None = None):
        if _token is not _NATIVE_TOKEN:
            raise TypeError(
                "OEFPMappingSet objects are created by fingerprint mapping factories."
            )
        self._native = native

    @classmethod
    def _from_native(cls, native: Any) -> OEFPMappingSet:
        return cls(native, _token=_NATIVE_TOKEN)

    def bit_ids(self, row: int = 0) -> tuple[int, ...]:
        """Return mapped bit ids for a fingerprint row."""
        return tuple(int(bit_id) for bit_id in self._native.BitIds32(int(row)))

    def atoms_for_bit(self, bit_id: int, row: int = 0) -> tuple[int, ...]:
        """Return center atom ids recorded for one bit."""
        return tuple(int(atom_id) for atom_id in self._native.AtomsForBit(int(row), int(bit_id)))

    def environments_for_bit(self, bit_id: int, row: int = 0) -> tuple[tuple[int, int], ...]:
        """Return ``(atom_id, radius)`` environments recorded for one bit."""
        atom_ids = self._native.EnvironmentAtomIdsForBit(int(row), int(bit_id))
        radii = self._native.EnvironmentRadiiForBit(int(row), int(bit_id))
        return tuple((int(atom_id), int(radius)) for atom_id, radius in zip(atom_ids, radii, strict=True))

    def bit_info(self, row: int = 0) -> dict[int, tuple[tuple[int, int], ...]]:
        """Return RDKit-style bit-info mappings for one fingerprint row."""
        return {
            bit_id: self.environments_for_bit(bit_id, row)
            for bit_id in self.bit_ids(row)
        }


class OEFPBatch:
    """Python wrapper for a native dense-binary OEFPBatch."""

    def __init__(self, native: Any, *, _token: object | None = None):
        if _token is not _NATIVE_TOKEN:
            raise TypeError(
                "OEFPBatch objects are created by OEFPBatch.from_fingerprints()."
            )
        self._native = native

    @classmethod
    def _from_native(cls, native: Any) -> OEFPBatch:
        return cls(native, _token=_NATIVE_TOKEN)

    @classmethod
    def from_fingerprints(cls, fingerprints: Sequence[OEFP]) -> OEFPBatch:
        """Create a contiguous batch from dense binary fingerprints."""
        native_fps = _native.OEFPVector()
        for fp in fingerprints:
            native_fps.push_back(fp._native)
        return cls._from_native(_native._NativeOEFPBatch.FromFingerprints(native_fps))

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

    @property
    def spec(self) -> FingerprintSpec:
        """Read-only fingerprint metadata for all rows."""
        return _fingerprint_spec(self._native.Spec())


class OEFPCountBatch:
    """Python wrapper for a native sparse counted OEFPCountBatch."""

    def __init__(self, native: Any, *, _token: object | None = None):
        if _token is not _NATIVE_TOKEN:
            raise TypeError(
                "OEFPCountBatch objects are created by OEFPCountBatch.from_fingerprints()."
            )
        self._native = native

    @classmethod
    def _from_native(cls, native: Any) -> OEFPCountBatch:
        return cls(native, _token=_NATIVE_TOKEN)

    @classmethod
    def from_fingerprints(cls, fingerprints: Sequence[OEFPCount]) -> OEFPCountBatch:
        """Create a contiguous sparse batch from counted fingerprints."""
        native_fps = _native.OEFPCountVector()
        for fp in fingerprints:
            native_fps.push_back(fp._native)
        return cls._from_native(_native._NativeOEFPCountBatch.FromFingerprints(native_fps))

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

    @property
    def spec(self) -> FingerprintSpec:
        """Read-only fingerprint metadata for all rows."""
        return _fingerprint_spec(self._native.Spec())


class OEFPSparseBatch:
    """Python wrapper for a native sparse binary OEFPSparseBatch."""

    def __init__(self, native: Any, *, _token: object | None = None):
        if _token is not _NATIVE_TOKEN:
            raise TypeError(
                "OEFPSparseBatch objects are created by OEFPSparseBatch.from_fingerprints()."
            )
        self._native = native

    @classmethod
    def _from_native(cls, native: Any) -> OEFPSparseBatch:
        return cls(native, _token=_NATIVE_TOKEN)

    @classmethod
    def from_fingerprints(cls, fingerprints: Sequence[OEFPSparse]) -> OEFPSparseBatch:
        """Create a contiguous sparse batch from sparse binary fingerprints."""
        native_fps = _native.OEFPSparseVector()
        for fp in fingerprints:
            native_fps.push_back(fp._native)
        return cls._from_native(_native._NativeOEFPSparseBatch.FromFingerprints(native_fps))

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
    def offsets(self) -> np.ndarray:
        """Read-only view of row offsets into indices."""
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
        """Number of flattened sparse binary entries."""
        return int(self._native.EntryCount())

    @property
    def num_bits(self) -> int:
        """Sparse fingerprint identifier domain size."""
        return int(self._native.SizeBits())

    @property
    def spec(self) -> FingerprintSpec:
        """Read-only fingerprint metadata for all rows."""
        return _fingerprint_spec(self._native.Spec())


@dataclass(frozen=True, init=False)
class Metric:
    """Python wrapper for native OEFP metrics."""

    _native: Any

    def __init__(self, native: Any, *, _token: object | None = None):
        if _token is not _NATIVE_TOKEN:
            raise TypeError(
                "Metric objects are created by Metric factory methods such as "
                "Metric.euclidean(), Metric.jaccard(), or Metric.tanimoto()."
            )
        object.__setattr__(self, "_native", native)

    @classmethod
    def _from_native(cls, native: Any) -> Metric:
        return cls(native, _token=_NATIVE_TOKEN)

    @property
    def name(self) -> str:
        """Metric formula name."""
        return _metric_name_name(self._native.Name())

    @property
    def type(self) -> str:
        """Whether the metric returns distance or similarity values."""
        return _metric_type_name(self._native.Type())

    @property
    def space(self) -> str:
        """Vector space the metric is intended to compare."""
        return _metric_space_name(self._native.Space())

    @property
    def p(self) -> float:
        """Minkowski exponent."""
        return float(self._native.P())

    @property
    def alpha(self) -> float:
        """Tversky alpha parameter."""
        return float(self._native.Alpha())

    @property
    def beta(self) -> float:
        """Tversky beta parameter."""
        return float(self._native.Beta())

    @property
    def weights(self) -> tuple[float, ...]:
        """Weighted Minkowski weights."""
        return tuple(float(value) for value in self._native.Weights())

    @property
    def variances(self) -> tuple[float, ...]:
        """Standardized Euclidean variances."""
        return tuple(float(value) for value in self._native.Variances())

    @property
    def inverse_covariance(self) -> tuple[float, ...]:
        """Mahalanobis inverse covariance matrix in row-major order."""
        return tuple(float(value) for value in self._native.InverseCovariance())

    @classmethod
    def euclidean(cls) -> Metric:
        """Create a Euclidean distance metric."""
        return cls._from_native(_native._NativeMetric.Euclidean())

    @classmethod
    def manhattan(cls) -> Metric:
        """Create a Manhattan distance metric."""
        return cls._from_native(_native._NativeMetric.Manhattan())

    @classmethod
    def chebyshev(cls) -> Metric:
        """Create a Chebyshev distance metric."""
        return cls._from_native(_native._NativeMetric.Chebyshev())

    @classmethod
    def minkowski(cls, p: float = 2.0, weights: Sequence[float] | None = None) -> Metric:
        """Create a Minkowski distance metric."""
        if weights is None:
            return cls._from_native(_native._NativeMetric.Minkowski(float(p)))
        return cls._from_native(_native._NativeMetric.Minkowski(float(p), _native_double_vector(weights)))

    @classmethod
    def standardized_euclidean(cls, variances: Sequence[float]) -> Metric:
        """Create a standardized Euclidean distance metric."""
        return cls._from_native(_native._NativeMetric.StandardizedEuclidean(_native_double_vector(variances)))

    @classmethod
    def seuclidean(cls, variances: Sequence[float]) -> Metric:
        """Create a standardized Euclidean distance metric."""
        return cls.standardized_euclidean(variances)

    @classmethod
    def mahalanobis(cls, inverse_covariance: Sequence[float]) -> Metric:
        """Create a Mahalanobis distance metric from an inverse covariance matrix."""
        return cls._from_native(_native._NativeMetric.Mahalanobis(_native_double_vector(inverse_covariance)))

    @classmethod
    def haversine(cls) -> Metric:
        """Create a Haversine distance metric."""
        return cls._from_native(_native._NativeMetric.Haversine())

    @classmethod
    def hamming(cls) -> Metric:
        """Create a Hamming distance metric."""
        return cls._from_native(_native._NativeMetric.Hamming())

    @classmethod
    def canberra(cls) -> Metric:
        """Create a Canberra distance metric."""
        return cls._from_native(_native._NativeMetric.Canberra())

    @classmethod
    def bray_curtis(cls) -> Metric:
        """Create a Bray-Curtis distance metric."""
        return cls._from_native(_native._NativeMetric.BrayCurtis())

    @classmethod
    def jaccard(cls) -> Metric:
        """Create a Jaccard distance metric."""
        return cls._from_native(_native._NativeMetric.Jaccard())

    @classmethod
    def matching(cls) -> Metric:
        """Create a Matching distance metric."""
        return cls._from_native(_native._NativeMetric.Matching())

    @classmethod
    def dice(cls) -> Metric:
        """Create a Dice distance metric."""
        return cls._from_native(_native._NativeMetric.Dice())

    @classmethod
    def kulsinski(cls) -> Metric:
        """Create a Kulsinski distance metric."""
        return cls._from_native(_native._NativeMetric.Kulsinski())

    @classmethod
    def rogers_tanimoto(cls) -> Metric:
        """Create a Rogers-Tanimoto distance metric."""
        return cls._from_native(_native._NativeMetric.RogersTanimoto())

    @classmethod
    def russell_rao(cls) -> Metric:
        """Create a Russell-Rao distance metric."""
        return cls._from_native(_native._NativeMetric.RussellRao())

    @classmethod
    def sokal_michener(cls) -> Metric:
        """Create a Sokal-Michener distance metric."""
        return cls._from_native(_native._NativeMetric.SokalMichener())

    @classmethod
    def sokal_sneath(cls) -> Metric:
        """Create a Sokal-Sneath distance metric."""
        return cls._from_native(_native._NativeMetric.SokalSneath())

    @classmethod
    def tanimoto(cls) -> Metric:
        """Create a Tanimoto similarity metric."""
        return cls._from_native(_native._NativeMetric.Tanimoto())

    @classmethod
    def tversky(
        cls,
        alpha: float,
        beta: float,
    ) -> Metric:
        """Create a Tversky similarity metric."""
        return cls._from_native(_native._NativeMetric.Tversky(alpha, beta))


class MorganGenerator:
    """Reusable generator for folded binary Morgan fingerprints."""

    def __init__(
        self,
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
    ) -> None:
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
        self._native = _native._NativeMorganGenerator(options)

    def fingerprint(self, mol: Any) -> OEFP:
        """Generate a folded dense binary Morgan fingerprint."""
        return OEFP._from_native(self._native.Fingerprint(mol))


class AtomPairGenerator:
    """Reusable generator for folded binary Atom Pair fingerprints."""

    def __init__(
        self,
        *,
        min_distance: int = 1,
        max_distance: int = 30,
        num_bits: int = 2048,
        use_chirality: bool = False,
        use_2d: bool = True,
        count_simulation: bool = True,
        count_bounds: Sequence[int] | None = None,
    ) -> None:
        options = _atom_pair_options(
            min_distance,
            max_distance,
            num_bits,
            use_chirality,
            use_2d,
            count_simulation,
            count_bounds,
        )
        self._native = _native._NativeAtomPairGenerator(options)

    def fingerprint(self, mol: Any) -> OEFP:
        """Generate a folded dense binary Atom Pair fingerprint."""
        return OEFP._from_native(self._native.Fingerprint(mol))


@dataclass(frozen=True)
class MorganFingerprintResult:
    """Dense Morgan fingerprint plus generated bit mappings."""

    fingerprint: OEFP
    mapping: OEFPMappingSet


@dataclass(frozen=True)
class MorganSparseFingerprintResult:
    """Sparse Morgan fingerprint plus generated bit mappings."""

    fingerprint: OEFPSparse
    mapping: OEFPMappingSet


@dataclass(frozen=True)
class MorganCountFingerprintResult:
    """Folded count Morgan fingerprint plus generated bit mappings."""

    fingerprint: OEFPCount
    mapping: OEFPMappingSet


@dataclass(frozen=True)
class MorganSparseCountFingerprintResult:
    """Sparse count Morgan fingerprint plus generated bit mappings."""

    fingerprint: OEFPCount
    mapping: OEFPMappingSet


def compare(
    a: OEFP | OEFPCount | OEFPSparse | DescriptorSet,
    b: (
        OEFP
        | OEFPBatch
        | OEFPCount
        | OEFPCountBatch
        | OEFPSparse
        | OEFPSparseBatch
        | DescriptorSet
        | DescriptorBatch
    ),
    metric: Metric,
    *,
    descriptor_mode: str = "count_overlap",
    num_threads: int = 0,
    chunk_size: int = 256,
) -> float | np.ndarray:
    """Compare one fingerprint or descriptor set with another object or batch."""
    if isinstance(a, OEFP) and isinstance(b, OEFP):
        return float(_native.Compare(a._native, b._native, metric._native))
    if isinstance(a, OEFPCount) and isinstance(b, OEFPCount):
        return float(_native.Compare(a._native, b._native, metric._native))
    if isinstance(a, OEFPSparse) and isinstance(b, OEFPSparse):
        return float(_native.Compare(a._native, b._native, metric._native))
    if isinstance(a, DescriptorSet) and isinstance(b, DescriptorSet):
        return float(
            _native.Compare(
                a._native,
                b._native,
                metric._native,
                _descriptor_mode_value(descriptor_mode),
            )
        )
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
    if isinstance(a, OEFPSparse) and isinstance(b, OEFPSparseBatch):
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
    if isinstance(a, DescriptorSet) and isinstance(b, DescriptorBatch):
        output = np.empty((b.size,), dtype=np.float64)
        _native.CompareIntoAddress(
            a._native,
            b._native,
            metric._native,
            _descriptor_mode_value(descriptor_mode),
            int(output.ctypes.data),
            output.size,
            _batch_options(num_threads, chunk_size),
        )
        return output

    raise TypeError(
        "compare expects OEFP/OEFP, OEFP/OEFPBatch, OEFPCount/OEFPCount, "
        "OEFPCount/OEFPCountBatch, OEFPSparse/OEFPSparse, or "
        "OEFPSparse/OEFPSparseBatch, DescriptorSet/DescriptorSet, or "
        "DescriptorSet/DescriptorBatch inputs."
    )


def cdist(
    a: OEFPBatch | OEFPCountBatch | OEFPSparseBatch | DescriptorBatch,
    b: OEFPBatch | OEFPCountBatch | OEFPSparseBatch | DescriptorBatch,
    metric: Metric,
    *,
    descriptor_mode: str = "count_overlap",
    num_threads: int = 0,
    chunk_size: int = 256,
) -> np.ndarray:
    """Return row-major cross-distance/comparison values as a 2D array."""
    if not (
        (isinstance(a, OEFPBatch) and isinstance(b, OEFPBatch))
        or (isinstance(a, OEFPCountBatch) and isinstance(b, OEFPCountBatch))
        or (isinstance(a, OEFPSparseBatch) and isinstance(b, OEFPSparseBatch))
        or (isinstance(a, DescriptorBatch) and isinstance(b, DescriptorBatch))
    ):
        raise TypeError(
            "cdist expects OEFPBatch/OEFPBatch, OEFPCountBatch/OEFPCountBatch, "
            "OEFPSparseBatch/OEFPSparseBatch, or DescriptorBatch/DescriptorBatch "
            "inputs."
        )

    output = np.empty((a.size, b.size), dtype=np.float64)
    if isinstance(a, DescriptorBatch) and isinstance(b, DescriptorBatch):
        _native.CDistIntoAddress(
            a._native,
            b._native,
            metric._native,
            _descriptor_mode_value(descriptor_mode),
            int(output.ctypes.data),
            output.size,
            _batch_options(num_threads, chunk_size),
        )
    else:
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
    batch: OEFPBatch | OEFPCountBatch | OEFPSparseBatch | DescriptorBatch,
    metric: Metric,
    *,
    descriptor_mode: str = "count_overlap",
    num_threads: int = 0,
    chunk_size: int = 256,
) -> np.ndarray:
    """Return SciPy-compatible condensed pairwise values."""
    if not isinstance(batch, (OEFPBatch, OEFPCountBatch, OEFPSparseBatch, DescriptorBatch)):
        raise TypeError(
            "pdist expects an OEFPBatch, OEFPCountBatch, OEFPSparseBatch, "
            "or DescriptorBatch input."
        )

    output = np.empty((batch.size * (batch.size - 1) // 2,), dtype=np.float64)
    if isinstance(batch, DescriptorBatch):
        _native.PDistIntoAddress(
            batch._native,
            metric._native,
            _descriptor_mode_value(descriptor_mode),
            int(output.ctypes.data),
            output.size,
            _batch_options(num_threads, chunk_size),
        )
    else:
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
    (
        radius_int,
        num_bits_int,
        use_chirality_bool,
        use_bond_types_bool,
        only_nonzero_invariants_bool,
        include_ring_membership_bool,
        include_redundant_environments_bool,
        count_simulation_bool,
        normalized_count_bounds,
    ) = _normalized_morgan_values(
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
    options = _native.MorganOptions()
    options.radius = radius_int
    options.num_bits = num_bits_int
    options.use_chirality = use_chirality_bool
    options.use_bond_types = use_bond_types_bool
    options.only_nonzero_invariants = only_nonzero_invariants_bool
    options.include_ring_membership = include_ring_membership_bool
    options.include_redundant_environments = include_redundant_environments_bool
    options.count_simulation = count_simulation_bool
    options.count_bounds = _native_uint32_vector(normalized_count_bounds)
    return options


def _normalized_morgan_values(
    radius: int,
    num_bits: int,
    use_chirality: bool,
    use_bond_types: bool,
    only_nonzero_invariants: bool,
    include_ring_membership: bool,
    include_redundant_environments: bool,
    count_simulation: bool,
    count_bounds: Sequence[int] | None,
) -> tuple[int, int, bool, bool, bool, bool, bool, bool, tuple[int, ...]]:
    radius_int = _uint32_option("Morgan", "radius", radius, positive=False)
    num_bits_int = _uint32_option("Morgan", "num_bits", num_bits, positive=True)
    normalized_count_bounds = _normalized_count_bounds("Morgan", count_bounds)
    if count_simulation and not normalized_count_bounds:
        raise ValueError("Morgan count_bounds cannot be empty when count simulation is enabled.")
    if count_simulation and len(normalized_count_bounds) >= num_bits_int:
        raise ValueError("Morgan count_bounds length must be smaller than num_bits.")
    if use_chirality:
        raise ValueError("Morgan chirality conformance is not implemented yet.")
    return (
        radius_int,
        num_bits_int,
        bool(use_chirality),
        bool(use_bond_types),
        bool(only_nonzero_invariants),
        bool(include_ring_membership),
        bool(include_redundant_environments),
        bool(count_simulation),
        normalized_count_bounds,
    )


@lru_cache(maxsize=32)
def _cached_morgan_generator(
    radius: int,
    num_bits: int,
    use_chirality: bool,
    use_bond_types: bool,
    only_nonzero_invariants: bool,
    include_ring_membership: bool,
    include_redundant_environments: bool,
    count_simulation: bool,
    count_bounds: tuple[int, ...],
) -> MorganGenerator:
    return MorganGenerator(
        radius=radius,
        num_bits=num_bits,
        use_chirality=use_chirality,
        use_bond_types=use_bond_types,
        only_nonzero_invariants=only_nonzero_invariants,
        include_ring_membership=include_ring_membership,
        include_redundant_environments=include_redundant_environments,
        count_simulation=count_simulation,
        count_bounds=count_bounds,
    )


@lru_cache(maxsize=32)
def _cached_atom_pair_generator(
    min_distance: int,
    max_distance: int,
    num_bits: int,
    use_chirality: bool,
    use_2d: bool,
    count_simulation: bool,
    count_bounds: tuple[int, ...],
) -> AtomPairGenerator:
    return AtomPairGenerator(
        min_distance=min_distance,
        max_distance=max_distance,
        num_bits=num_bits,
        use_chirality=use_chirality,
        use_2d=use_2d,
        count_simulation=count_simulation,
        count_bounds=count_bounds,
    )


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
    generator = _cached_morgan_generator(
        *_normalized_morgan_values(
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
    )
    return generator.fingerprint(mol)


def atom_pair_fingerprint(
    mol: Any,
    *,
    min_distance: int = 1,
    max_distance: int = 30,
    num_bits: int = 2048,
    use_chirality: bool = False,
    use_2d: bool = True,
    count_simulation: bool = True,
    count_bounds: Sequence[int] | None = None,
) -> OEFP:
    """Generate an RDKit-compatible folded binary Atom Pair fingerprint."""
    generator = _cached_atom_pair_generator(
        *_normalized_atom_pair_values(
            min_distance,
            max_distance,
            num_bits,
            use_chirality,
            use_2d,
            count_simulation,
            count_bounds,
        )
    )
    return generator.fingerprint(mol)


def atom_pair_count_fingerprint(
    mol: Any,
    *,
    min_distance: int = 1,
    max_distance: int = 30,
    num_bits: int = 2048,
    use_chirality: bool = False,
    use_2d: bool = True,
) -> OEFPCount:
    """Generate an RDKit-compatible folded count Atom Pair fingerprint."""
    options = _atom_pair_options(
        min_distance,
        max_distance,
        num_bits,
        use_chirality,
        use_2d,
        False,
        None,
    )
    return OEFPCount._from_native(_native.MakeAtomPairCountFingerprint(mol, options))


def atom_pair_sparse_fingerprint(
    mol: Any,
    *,
    min_distance: int = 1,
    max_distance: int = 30,
    use_chirality: bool = False,
    use_2d: bool = True,
    count_simulation: bool = True,
    count_bounds: Sequence[int] | None = None,
) -> OEFPSparse:
    """Generate an RDKit-compatible sparse binary Atom Pair fingerprint."""
    options = _atom_pair_options(
        min_distance,
        max_distance,
        1 << 23,
        use_chirality,
        use_2d,
        count_simulation,
        count_bounds,
    )
    return OEFPSparse._from_native(_native.MakeAtomPairSparseFingerprint(mol, options))


def atom_pair_sparse_count_fingerprint(
    mol: Any,
    *,
    min_distance: int = 1,
    max_distance: int = 30,
    use_chirality: bool = False,
    use_2d: bool = True,
) -> OEFPCount:
    """Generate an RDKit-compatible sparse count Atom Pair fingerprint."""
    options = _atom_pair_options(
        min_distance,
        max_distance,
        1 << 23,
        use_chirality,
        use_2d,
        False,
        None,
    )
    return OEFPCount._from_native(_native.MakeAtomPairSparseCountFingerprint(mol, options))


def _atom_pair_descriptor_options(
    min_distance: int,
    max_distance: int,
    use_chirality: bool,
    use_2d: bool,
) -> Any:
    min_distance_int = _uint32_option("Atom Pair", "min_distance", min_distance, positive=False)
    max_distance_int = _uint32_option("Atom Pair", "max_distance", max_distance, positive=False)
    if min_distance_int > max_distance_int:
        raise ValueError("Atom Pair min_distance cannot exceed max_distance.")
    if max_distance_int >= 31:
        raise ValueError("Atom Pair max_distance must be smaller than 31.")
    if use_chirality:
        raise ValueError("Atom Pair chirality conformance is not implemented yet.")
    if not use_2d:
        raise ValueError("Atom Pair 3D distance conformance is not implemented yet.")

    options = _native.AtomPairOptions()
    options.min_distance = min_distance_int
    options.max_distance = max_distance_int
    options.use_chirality = bool(use_chirality)
    options.use_2d = bool(use_2d)
    options.count_simulation = False
    return options


def atom_pair_descriptors(
    mol: Any,
    *,
    min_distance: int = 1,
    max_distance: int = 30,
    use_chirality: bool = False,
    use_2d: bool = True,
) -> DescriptorSet:
    """Generate raw Atom Pair descriptors as counted string keys."""
    options = _atom_pair_descriptor_options(
        min_distance,
        max_distance,
        use_chirality,
        use_2d,
    )
    return DescriptorSet._from_native(_native.MakeAtomPairDescriptors(mol, options))


def morgan_descriptors(
    mol: Any,
    *,
    radius: int = 2,
    use_chirality: bool = False,
    use_bond_types: bool = True,
    only_nonzero_invariants: bool = False,
    include_ring_membership: bool = True,
    include_redundant_environments: bool = False,
    count_simulation: bool = False,
) -> DescriptorSet:
    """Generate raw Morgan descriptors as counted integer keys."""
    if count_simulation:
        raise ValueError("Morgan count simulation is only supported for binary fingerprints.")
    options = _morgan_options(
        radius,
        2048,
        use_chirality,
        use_bond_types,
        only_nonzero_invariants,
        include_ring_membership,
        include_redundant_environments,
    )
    return DescriptorSet._from_native(_native.MakeMorganDescriptors(mol, options))


def morgan_fingerprint_with_mapping(
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
) -> MorganFingerprintResult:
    """Generate a folded binary Morgan fingerprint and bit-info mapping."""
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
    native = _native.MakeMorganFingerprintWithMapping(mol, options)
    return MorganFingerprintResult(
        OEFP._from_native(native.Fingerprint()),
        OEFPMappingSet._from_native(native.Mapping()),
    )


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
    return OEFPCount._from_native(_native.MakeMorganCountFingerprint(mol, options))


def morgan_count_fingerprint_with_mapping(
    mol: Any,
    *,
    radius: int = 2,
    num_bits: int = 2048,
    use_chirality: bool = False,
    use_bond_types: bool = True,
    only_nonzero_invariants: bool = False,
    include_ring_membership: bool = True,
    include_redundant_environments: bool = False,
) -> MorganCountFingerprintResult:
    """Generate a folded count Morgan fingerprint and bit-info mapping."""
    options = _morgan_options(
        radius,
        num_bits,
        use_chirality,
        use_bond_types,
        only_nonzero_invariants,
        include_ring_membership,
        include_redundant_environments,
    )
    native = _native.MakeMorganCountFingerprintWithMapping(mol, options)
    return MorganCountFingerprintResult(
        OEFPCount._from_native(native.Fingerprint()),
        OEFPMappingSet._from_native(native.Mapping()),
    )


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
    return OEFPCount._from_native(_native.MakeMorganSparseCountFingerprint(mol, options))


def morgan_sparse_count_fingerprint_with_mapping(
    mol: Any,
    *,
    radius: int = 2,
    use_chirality: bool = False,
    use_bond_types: bool = True,
    only_nonzero_invariants: bool = False,
    include_ring_membership: bool = True,
    include_redundant_environments: bool = False,
) -> MorganSparseCountFingerprintResult:
    """Generate a sparse count Morgan fingerprint and raw bit-info mapping."""
    options = _morgan_options(
        radius,
        2048,
        use_chirality,
        use_bond_types,
        only_nonzero_invariants,
        include_ring_membership,
        include_redundant_environments,
    )
    native = _native.MakeMorganSparseCountFingerprintWithMapping(mol, options)
    return MorganSparseCountFingerprintResult(
        OEFPCount._from_native(native.Fingerprint()),
        OEFPMappingSet._from_native(native.Mapping()),
    )


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
    return OEFPSparse._from_native(_native.MakeMorganSparseFingerprint(mol, options))


def morgan_sparse_fingerprint_with_mapping(
    mol: Any,
    *,
    radius: int = 2,
    use_chirality: bool = False,
    use_bond_types: bool = True,
    only_nonzero_invariants: bool = False,
    include_ring_membership: bool = True,
    include_redundant_environments: bool = False,
) -> MorganSparseFingerprintResult:
    """Generate a sparse binary Morgan fingerprint and raw bit-info mapping."""
    options = _morgan_options(
        radius,
        2048,
        use_chirality,
        use_bond_types,
        only_nonzero_invariants,
        include_ring_membership,
        include_redundant_environments,
    )
    native = _native.MakeMorganSparseFingerprintWithMapping(mol, options)
    return MorganSparseFingerprintResult(
        OEFPSparse._from_native(native.Fingerprint()),
        OEFPMappingSet._from_native(native.Mapping()),
    )


def from_openeye_fingerprint(fp: Any) -> OEFP:
    """Import an OpenEye OEFingerPrint as an OEFP."""
    return OEFP._from_native(_native.FromOEFingerPrint(fp))


def to_openeye_fingerprint(fp: OEFP) -> Any:
    """Export an OEFP as an OpenEye OEFingerPrint."""
    from openeye import oechem, oegraphsim

    spec = fp._native.Spec()
    if (
        spec.value_type != _native.FingerprintValueType_Binary
        or spec.source_name != "OpenEye"
    ):
        raise ValueError("OEFP spec does not contain OpenEye fingerprint type metadata.")

    fp_type = None
    if spec.source_type:
        fp_type = oegraphsim.OEGetFPType(spec.source_type)
        if fp_type is None:
            raise ValueError("OpenEye fingerprint type metadata could not be resolved.")
    elif spec.has_source_type_id:
        fp_type = oegraphsim.OEGetFPType(spec.source_type_id)

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
