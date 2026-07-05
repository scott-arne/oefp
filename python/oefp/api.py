"""Public Python API for OEFP."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from . import _native
from ._views import readonly_array_from_address

_UINT32_MAX = 2**32 - 1
DESCRIPTOR_PREREQUISITE_NONE = 0
DESCRIPTOR_PREREQUISITE_GRAPH = 1 << 0
DESCRIPTOR_PREREQUISITE_COORDINATES_2D = 1 << 1
DESCRIPTOR_PREREQUISITE_COORDINATES_3D = 1 << 2
DESCRIPTOR_PREREQUISITE_ALL = _UINT32_MAX
TOPOLOGICAL_ATOM_PAIR_PREREQUISITES = DESCRIPTOR_PREREQUISITE_GRAPH
TOPOLOGICAL_TORSIONS_PREREQUISITES = DESCRIPTOR_PREREQUISITE_GRAPH
DISTANCE_ATOM_PAIR_PREREQUISITES = (
    DESCRIPTOR_PREREQUISITE_GRAPH | DESCRIPTOR_PREREQUISITE_COORDINATES_3D
)
_NATIVE_TOKEN = object()
_DESCRIPTOR_SCHEMA_METADATA_KEY = b"oefp.python_descriptor_schema"
_ROW_IDS_METADATA_KEY = b"oefp.row_ids_json"
_MORDRED_REFERENCE_RESOURCE = "mordred_references.json"
_MORDRED_REFERENCE_FIXTURE = (
    Path(__file__).resolve().parents[2] / "tests" / "python" / "mordred_references.json"
)


def _normalized_descriptor_prerequisites(value: int) -> int:
    prerequisites = int(value)
    if prerequisites < 0 or prerequisites > _UINT32_MAX:
        raise ValueError("Descriptor prerequisites must fit in uint32.")
    return prerequisites


def descriptor_missing_prerequisites(required: int, available: int) -> int:
    """Return prerequisite bits required by a descriptor but absent from an input.

    :param required: Descriptor prerequisite bitmap.
    :param available: Input prerequisite bitmap.
    :returns: Bitmap of missing prerequisite bits.
    """
    required_bits = _normalized_descriptor_prerequisites(required)
    available_bits = _normalized_descriptor_prerequisites(available)
    return required_bits & (~available_bits & _UINT32_MAX)


def descriptor_prerequisites_satisfied(required: int, available: int) -> bool:
    """Return whether an input satisfies a descriptor prerequisite bitmap.

    :param required: Descriptor prerequisite bitmap.
    :param available: Input prerequisite bitmap.
    :returns: ``True`` when all required bits are present.
    """
    return descriptor_missing_prerequisites(required, available) == DESCRIPTOR_PREREQUISITE_NONE


def _molecule_has_3d_coordinates(mol: Any) -> bool:
    get_dimension = getattr(mol, "GetDimension", None)
    if get_dimension is None or int(get_dimension()) != 3:
        return False

    num_atoms_fn = getattr(mol, "NumAtoms", None)
    expected_atoms = int(num_atoms_fn()) if num_atoms_fn is not None else 0
    if expected_atoms == 0:
        return False

    get_coords = getattr(mol, "GetCoords", None)
    if get_coords is None:
        return False
    coords = get_coords()
    if not isinstance(coords, Mapping) or len(coords) < expected_atoms:
        return False
    return all(len(tuple(value)) >= 3 for value in coords.values())


def _require_distance_atom_pair_3d(mol: Any) -> None:
    if not _molecule_has_3d_coordinates(mol):
        raise ValueError(
            "Distance Atom Pair requires existing 3D coordinates; "
            "OEFP does not generate conformers implicitly."
        )


def _raise_distance_atom_pair_not_implemented() -> NoReturn:
    raise NotImplementedError(
        "Distance Atom Pair requires existing 3D coordinates and is not implemented yet."
    )


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


@dataclass(frozen=True)
class DescriptorDefinition:
    """Descriptor column metadata for schema-backed descriptor rows.

    :param name: Descriptor column name.
    :param value_type: Scalar value type, such as ``"float"``, ``"int"``,
        ``"bool"``, or ``"string"``.
    :param group: Optional descriptor group used for selection.
    :param source_name: Optional source package or toolkit name.
    :param source_type: Optional source descriptor family.
    :param source_version: Optional source version string.
    :param parameters: Optional serialized source parameters.
    :param units: Optional physical units for numeric descriptors.
    :param description: Optional human-readable descriptor description.
    :param prerequisites: Bitmap of input prerequisites needed to compute
        this descriptor.
    :param canonical_id: Optional curated cross-source identity. A non-empty
        namespaced string (e.g. ``"quantity:exact_molecular_weight"``) marks
        descriptors that run the same computation and therefore produce
        provably identical values. Empty means the descriptor has no known
        cross-source equivalent and is never deduplicated.
    """

    name: str
    value_type: str
    group: str = ""
    source_name: str = ""
    source_type: str = ""
    source_version: str = ""
    parameters: str = ""
    units: str = ""
    description: str = ""
    prerequisites: int = 0
    canonical_id: str = ""

    def __post_init__(self) -> None:
        normalized = _normalized_descriptor_kind(self.value_type)
        prerequisites = _normalized_descriptor_prerequisites(self.prerequisites)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "value_type", normalized)
        object.__setattr__(self, "group", str(self.group))
        object.__setattr__(self, "source_name", str(self.source_name))
        object.__setattr__(self, "source_type", str(self.source_type))
        object.__setattr__(self, "source_version", str(self.source_version))
        object.__setattr__(self, "parameters", str(self.parameters))
        object.__setattr__(self, "units", str(self.units))
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "prerequisites", prerequisites)
        object.__setattr__(self, "canonical_id", str(self.canonical_id))

    def _metadata(self) -> dict[str, str | int]:
        return {
            "name": self.name,
            "value_type": self.value_type,
            "group": self.group,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "source_version": self.source_version,
            "parameters": self.parameters,
            "units": self.units,
            "description": self.description,
            "canonical_id": self.canonical_id,
            "prerequisites": self.prerequisites,
        }

    @classmethod
    def _from_metadata(cls, metadata: Mapping[str, Any]) -> DescriptorDefinition:
        return cls(
            str(metadata["name"]),
            str(metadata["value_type"]),
            group=str(metadata.get("group", "")),
            source_name=str(metadata.get("source_name", "")),
            source_type=str(metadata.get("source_type", "")),
            source_version=str(metadata.get("source_version", "")),
            parameters=str(metadata.get("parameters", "")),
            units=str(metadata.get("units", "")),
            description=str(metadata.get("description", "")),
            prerequisites=int(metadata.get("prerequisites", 0)),
            canonical_id=str(metadata.get("canonical_id", "")),
        )


class DescriptorSchema:
    """Schema for named descriptor columns.

    :param definitions: Descriptor column definitions in storage order.
    :raises ValueError: When descriptor names are empty or duplicated.
    """

    def __init__(self, definitions: Sequence[DescriptorDefinition]):
        normalized = tuple(definitions)
        if not normalized:
            raise ValueError("DescriptorSchema requires at least one definition.")
        names = tuple(definition.name for definition in normalized)
        if any(not name for name in names):
            raise ValueError("Descriptor schema names must be non-empty.")
        if len(set(names)) != len(names):
            raise ValueError("Descriptor schema names must be unique.")
        self._definitions = normalized
        self._index_by_name = {name: index for index, name in enumerate(names)}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DescriptorSchema):
            return NotImplemented
        return self.definitions == other.definitions

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __repr__(self) -> str:
        return f"DescriptorSchema(columns={len(self._definitions)})"

    def __len__(self) -> int:
        return len(self._definitions)

    @property
    def definitions(self) -> tuple[DescriptorDefinition, ...]:
        """Descriptor definitions in schema order."""
        return self._definitions

    @property
    def schema_id(self) -> str:
        """Stable identifier derived from descriptor definitions."""
        return hashlib.sha256(self._metadata().encode("utf-8")).hexdigest()

    @property
    def names(self) -> tuple[str, ...]:
        """Descriptor names in schema order."""
        return tuple(definition.name for definition in self._definitions)

    def index(self, name: str) -> int:
        """Return the integer position for a descriptor name.

        :param name: Descriptor name to resolve.
        :returns: Zero-based descriptor position.
        :raises KeyError: When the descriptor name is not present.
        """
        return self._index_by_name[name]

    def group(self, group: str) -> tuple[int, ...]:
        """Return descriptor positions that belong to a group.

        :param group: Descriptor group label.
        :returns: Tuple of descriptor positions in schema order.
        """
        return tuple(
            index
            for index, definition in enumerate(self._definitions)
            if definition.group == group
        )

    def subset(self, names: Sequence[str]) -> DescriptorSchema:
        """Return a schema projected to descriptor names in selection order.

        :param names: Descriptor names to keep.
        :returns: Projected descriptor schema.
        """
        return DescriptorSchema([self._definitions[self.index(name)] for name in names])

    def _metadata(self) -> str:
        return json.dumps(
            [definition._metadata() for definition in self._definitions],
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def _from_metadata(cls, metadata: bytes) -> DescriptorSchema:
        raw_definitions = json.loads(metadata.decode("utf-8"))
        if not isinstance(raw_definitions, list):
            raise ValueError("Descriptor schema metadata must contain a list.")
        # Reconstruct symmetrically with serialization: a zero-column schema is a
        # supported (empty calculator) state, so bypass the public empty check.
        return cls._allow_empty(
            [DescriptorDefinition._from_metadata(item) for item in raw_definitions]
        )

    @classmethod
    def _allow_empty(cls, definitions: Sequence[DescriptorDefinition]) -> DescriptorSchema:
        """Construct a schema that may be empty (calculator merged-schema path only).

        The public constructor rejects an empty schema, but a
        :class:`DescriptorCalculator` may legitimately resolve to zero columns
        (no sources, or every column selected or deduplicated away). This
        bypasses only the empty check; non-empty names and uniqueness are still
        enforced.

        :param definitions: Descriptor column definitions in storage order.
        :returns: Descriptor schema that may contain zero definitions.
        :raises ValueError: When descriptor names are empty or duplicated.
        """
        obj = cls.__new__(cls)
        normalized = tuple(definitions)
        names = tuple(definition.name for definition in normalized)
        if any(not name for name in names):
            raise ValueError("Descriptor schema names must be non-empty.")
        if len(set(names)) != len(names):
            raise ValueError("Descriptor schema names must be unique.")
        obj._definitions = normalized
        obj._index_by_name = {name: index for index, name in enumerate(names)}
        return obj


def _value_type_name(value: Any) -> str:
    if value == _native.FingerprintValueType_Binary:
        return "binary"
    if value == _native.FingerprintValueType_Counted:
        return "counted"
    return "unknown"


def _normalized_descriptor_kind(value_type: str) -> str:
    normalized = str(value_type).lower()
    if normalized == "integer":
        return "int"
    if normalized in {"bool", "int", "float", "string"}:
        return normalized
    raise ValueError(f"Unknown descriptor value type: {value_type!r}.")


def _descriptor_arrow_type(pa: Any, value_type: str) -> Any:
    if value_type == "bool":
        return pa.bool_()
    if value_type == "int":
        return pa.int64()
    if value_type == "float":
        return pa.float64()
    if value_type == "string":
        return pa.string()
    raise ValueError(f"Unknown descriptor value type: {value_type!r}.")


def _is_schema_descriptor_set(value: DescriptorSet) -> bool:
    return value._native is None


def _is_schema_descriptor_batch(value: DescriptorBatch) -> bool:
    return value._native is None


def _raise_schema_descriptor_compare_error() -> None:
    raise TypeError(
        "compare, cdist, and pdist do not support schema-backed descriptors."
    )


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


def _descriptor_kind_name(value_kind: Any) -> str:
    if value_kind == _native.DescriptorValueKind_Bool:
        return "bool"
    if value_kind == _native.DescriptorValueKind_Int:
        return "int"
    if value_kind == _native.DescriptorValueKind_Float:
        return "float"
    if value_kind == _native.DescriptorValueKind_String:
        return "string"
    raise ValueError(f"Unsupported descriptor value kind: {value_kind!r}.")


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


@lru_cache(maxsize=1)
def _mordred_reference_payload() -> dict[str, Any]:
    resource = resources.files("oefp").joinpath(_MORDRED_REFERENCE_RESOURCE)
    if resource.is_file():
        with resource.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        with _MORDRED_REFERENCE_FIXTURE.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("definitions"), list):
        raise ValueError("Mordred reference fixture does not contain definitions.")
    return payload


def _mordred_definition_from_fixture(definition: Mapping[str, Any]) -> DescriptorDefinition:
    return DescriptorDefinition(
        str(definition["name"]),
        str(definition["value_kind"]),
        group=str(definition.get("group", "")),
        source_name=str(definition.get("source_name", "")),
        source_type=str(definition.get("source_type", "")),
        source_version=str(definition.get("source_version", "")),
        parameters=str(definition.get("parameters", "")),
        units=str(definition.get("units", "")),
        description=str(definition.get("description", "")),
        prerequisites=int(definition.get("prerequisites", 0)),
        canonical_id=str(definition.get("canonical_id", "")),
    )


@lru_cache(maxsize=1)
def mordred_schema() -> DescriptorSchema:
    """Return the full Mordred descriptor schema.

    :returns: Descriptor schema matching the committed Mordred 1.2.0 fixture.
    """
    payload = _mordred_reference_payload()
    return DescriptorSchema(
        [_mordred_definition_from_fixture(item) for item in payload["definitions"]]
    )


def _mordred_value_from_native(native: Any, definition: DescriptorDefinition) -> Any:
    if not native.Has(definition.name):
        return None
    if definition.value_type == "bool":
        return bool(native.Bool(definition.name))
    if definition.value_type == "int":
        return int(native.Int(definition.name))
    if definition.value_type == "float":
        return float(native.Float(definition.name))
    if definition.value_type == "string":
        return str(native.String(definition.name))
    return None


def _legacy_counted_string_descriptor(native: Any, spec: DescriptorSpec) -> DescriptorSet:
    return DescriptorSet._from_native(
        _native._NativeDescriptorSet.FromStringCounts(
            _native_descriptor_spec(spec),
            native.StringKeys(),
            native.Counts(),
        )
    )


def _legacy_counted_integer_descriptor(native: Any, spec: DescriptorSpec) -> DescriptorSet:
    return DescriptorSet._from_native(
        _native._NativeDescriptorSet.FromIntegerCounts(
            _native_descriptor_spec(spec),
            native.IntegerKeys(),
            native.Counts(),
        )
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
    if not use_2d:
        raise ValueError(
            "Distance Atom Pair requires existing 3D coordinates and is not implemented yet."
        )
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


def _normalized_topological_torsions_values(
    torsion_atom_count: int,
    num_bits: int,
    use_chirality: bool,
    count_simulation: bool,
    count_bounds: Sequence[int] | None,
) -> tuple[int, int, bool, bool, tuple[int, ...]]:
    torsion_atom_count_int = _uint32_option(
        "Topological Torsions",
        "torsion_atom_count",
        torsion_atom_count,
        positive=True,
    )
    num_bits_int = _uint32_option("Topological Torsions", "num_bits", num_bits, positive=True)
    normalized_count_bounds = _normalized_count_bounds("Topological Torsions", count_bounds)
    if torsion_atom_count_int >= 8:
        raise ValueError("Topological Torsions torsion_atom_count must be smaller than 8.")
    if count_simulation and not normalized_count_bounds:
        raise ValueError(
            "Topological Torsions count_bounds cannot be empty when count simulation is enabled."
        )
    if count_simulation and len(normalized_count_bounds) >= num_bits_int:
        raise ValueError(
            "Topological Torsions count_bounds length must be smaller than num_bits."
        )
    return (
        torsion_atom_count_int,
        num_bits_int,
        bool(use_chirality),
        bool(count_simulation),
        normalized_count_bounds,
    )


def _topological_torsions_options(
    torsion_atom_count: int,
    num_bits: int,
    use_chirality: bool,
    count_simulation: bool,
    count_bounds: Sequence[int] | None,
) -> Any:
    (
        torsion_atom_count_int,
        num_bits_int,
        use_chirality_bool,
        count_simulation_bool,
        normalized_count_bounds,
    ) = _normalized_topological_torsions_values(
        torsion_atom_count,
        num_bits,
        use_chirality,
        count_simulation,
        count_bounds,
    )
    options = _native.TopologicalTorsionsOptions()
    options.torsion_atom_count = torsion_atom_count_int
    options.num_bits = num_bits_int
    options.use_chirality = use_chirality_bool
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

    def __repr__(self) -> str:
        return (
            f"OEFP(num_bits={self.num_bits}, popcount={self.popcount}, "
            f"source_type={self.spec.source_type!r})"
        )

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

    def __repr__(self) -> str:
        # num_bits is the (possibly unfolded) identifier domain rather than a fold
        # size, so the occupancy counts are the relevant per-molecule data.
        return (
            f"OEFPCount(nonzero_count={self.nonzero_count}, "
            f"total_count={self.total_count}, source_type={self.spec.source_type!r})"
        )

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


class OEFPCount64:
    """Python wrapper for a native sparse counted fingerprint with uint64 indices."""

    def __init__(self, native: Any, *, _token: object | None = None):
        if _token is not _NATIVE_TOKEN:
            raise TypeError(
                "OEFPCount64 objects are created by OEFPCount64 factories such as "
                "topological_torsions_sparse_count_fingerprint()."
            )
        self._native = native

    @classmethod
    def _from_native(cls, native: Any) -> OEFPCount64:
        return cls(native, _token=_NATIVE_TOKEN)

    def __repr__(self) -> str:
        # num_bits is the raw identifier domain rather than a fold size, so the
        # occupancy counts are the relevant per-molecule data.
        return (
            f"OEFPCount64(nonzero_count={self.nonzero_count}, "
            f"total_count={self.total_count}, source_type={self.spec.source_type!r})"
        )

    @property
    def indices(self) -> np.ndarray:
        """Read-only view of sorted nonzero uint64 count indices."""
        return readonly_array_from_address(
            self,
            self._native.IndexDataAddress(),
            (self._native.NonzeroCount(),),
            np.dtype(np.uint64),
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
        """Raw fingerprint identifier domain size."""
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

    def __repr__(self) -> str:
        # num_bits is the unfolded identifier domain (often the full uint64
        # space), not a fold size, so popcount is the relevant per-molecule data.
        return f"OEFPSparse(popcount={self.popcount}, source_type={self.spec.source_type!r})"

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
    """Python wrapper for legacy native or schema-backed descriptor rows."""

    def __init__(
        self,
        schema: DescriptorSchema | Any,
        values: Mapping[str, Any] | None = None,
        *,
        row_id: str = "",
        _token: object | None = None,
    ):
        if _token is _NATIVE_TOKEN:
            self._native: Any = schema
            self._schema: DescriptorSchema | None = None
            self._values: dict[str, Any] | None = None
            self._row_id = ""
            return
        if not isinstance(schema, DescriptorSchema) or values is None:
            raise TypeError(
                "DescriptorSet objects are created by DescriptorSet(schema, values), "
                "DescriptorSet.from_strings(), DescriptorSet.from_integers(), "
                "DescriptorSet.from_floats(), or descriptor factories."
            )
        self._native = None
        self._schema = schema
        self._values = self._normalized_values(schema, values)
        self._row_id = str(row_id)

    @classmethod
    def _from_native(cls, native: Any) -> DescriptorSet:
        return cls(native, _token=_NATIVE_TOKEN)

    def __repr__(self) -> str:
        if self._native is None:
            columns = len(self._schema.definitions) if self._schema is not None else 0
            if self._row_id:
                return f"DescriptorSet(columns={columns}, row_id={self._row_id!r})"
            return f"DescriptorSet(columns={columns})"
        return (
            f"DescriptorSet(value_type={self.value_type!r}, size={self.size}, "
            f"total_count={self.total_count})"
        )

    @staticmethod
    def _normalized_values(
        schema: DescriptorSchema,
        values: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for definition in schema.definitions:
            value = values.get(definition.name)
            if value is None:
                normalized[definition.name] = None
            elif definition.value_type == "bool":
                if not isinstance(value, bool):
                    raise TypeError(f"Descriptor {definition.name!r} must be a bool.")
                normalized[definition.name] = value
            elif definition.value_type == "int":
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise TypeError(f"Descriptor {definition.name!r} must be an integer.")
                normalized[definition.name] = int(value)
            elif definition.value_type == "float":
                if isinstance(value, bool):
                    raise TypeError(f"Descriptor {definition.name!r} must be a float.")
                normalized[definition.name] = float(value)
            elif definition.value_type == "string":
                if not isinstance(value, str):
                    raise TypeError(f"Descriptor {definition.name!r} must be a string.")
                normalized[definition.name] = value
        return normalized

    def _require_native(self) -> Any:
        if self._native is None:
            raise TypeError(
                "Schema-backed descriptor sets do not expose legacy descriptor storage."
            )
        return self._native

    def _require_schema(self) -> DescriptorSchema:
        if self._schema is None:
            raise TypeError("Legacy descriptor sets do not expose a descriptor schema.")
        return self._schema

    def __getitem__(self, name: str) -> Any:
        """Return one schema-backed descriptor value by name.

        :param name: Descriptor name.
        :returns: Descriptor value.
        :raises TypeError: When the descriptor row is not schema-backed.
        """
        if self._values is None:
            raise TypeError("Legacy descriptor sets do not support named value access.")
        self._require_schema().index(name)
        return self._values[name]

    def subset(self, names: Sequence[str]) -> DescriptorSet:
        """Return a schema-backed descriptor row projected to named columns.

        :param names: Descriptor names to keep.
        :returns: Projected descriptor row.
        :raises TypeError: When the descriptor row is not schema-backed.
        """
        if self._values is None:
            raise TypeError("Legacy descriptor sets do not support named subsets.")
        schema = self.schema.subset(names)
        return DescriptorSet(
            schema,
            {name: self._values[name] for name in schema.names},
            row_id=self._row_id,
        )

    @property
    def schema(self) -> DescriptorSchema:
        """Descriptor schema for schema-backed rows."""
        return self._require_schema()

    @property
    def row_id(self) -> str:
        """Optional schema-backed row identifier."""
        return self._row_id

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
        return _descriptor_value_type_name(self._require_native().ValueType())

    @property
    def size(self) -> int:
        """Number of unique descriptor keys."""
        return int(self._require_native().Size())

    @property
    def total_count(self) -> int:
        """Sum of all descriptor counts."""
        return int(self._require_native().TotalCount())

    @property
    def string_keys(self) -> tuple[str, ...]:
        """Canonical sorted string keys."""
        if self.value_type != "string":
            return ()
        return tuple(str(value) for value in self._require_native().StringKeys())

    @property
    def integer_keys(self) -> tuple[int, ...]:
        """Canonical sorted integer keys."""
        if self.value_type != "integer":
            return ()
        keys = readonly_array_from_address(
            self,
            self._require_native().IntegerKeyDataAddress(),
            (self._require_native().Size(),),
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
            self._require_native().FloatKeyDataAddress(),
            (self._require_native().Size(),),
            np.dtype(np.float64),
        )
        return tuple(float(value) for value in keys)

    @property
    def counts(self) -> np.ndarray:
        """Read-only view of descriptor counts parallel to active keys."""
        return readonly_array_from_address(
            self,
            self._require_native().CountDataAddress(),
            (self._require_native().Size(),),
            np.dtype(np.uint32),
        )

    @property
    def spec(self) -> DescriptorSpec:
        """Read-only descriptor metadata."""
        return _descriptor_spec(self._require_native().Spec())


class DescriptorBatch:
    """Python wrapper for legacy native or schema-backed descriptor batches."""

    def __init__(
        self,
        native: Any | None = None,
        *,
        schema: DescriptorSchema | None = None,
        rows: Sequence[Mapping[str, Any]] | None = None,
        row_ids: Sequence[str] | None = None,
        _token: object | None = None,
    ):
        if _token is _NATIVE_TOKEN:
            self._native: Any = native
            self._schema: DescriptorSchema | None = None
            self._rows: tuple[dict[str, Any], ...] | None = None
            self._row_ids: tuple[str, ...] = ()
            return
        if schema is None or rows is None:
            raise TypeError(
                "DescriptorBatch objects are created by "
                "DescriptorBatch.from_descriptors()."
            )
        normalized_rows = tuple(
            DescriptorSet._normalized_values(schema, row)
            for row in rows
        )
        if row_ids is not None and len(row_ids) != len(normalized_rows):
            raise ValueError("DescriptorBatch row_ids length must match row count.")
        self._native = None
        self._schema = schema
        self._rows = normalized_rows
        self._row_ids = tuple(row_ids or ("",) * len(normalized_rows))

    @classmethod
    def _from_native(cls, native: Any) -> DescriptorBatch:
        return cls(native, _token=_NATIVE_TOKEN)

    @classmethod
    def from_descriptors(cls, descriptors: Sequence[DescriptorSet]) -> DescriptorBatch:
        """Create a contiguous descriptor batch from descriptor sets."""
        if not descriptors:
            return cls._from_native(_native._NativeDescriptorBatch.FromDescriptorSets(_native.DescriptorSetVector()))
        if all(descriptor._schema is not None for descriptor in descriptors):
            schema = descriptors[0].schema
            rows: list[Mapping[str, Any]] = []
            row_ids: list[str] = []
            for descriptor_set in descriptors:
                if descriptor_set.schema != schema:
                    raise ValueError("Descriptor set schema does not match batch schema.")
                if descriptor_set._values is None:
                    raise TypeError("Descriptor set is not schema-backed.")
                rows.append(descriptor_set._values)
                row_ids.append(descriptor_set.row_id)
            return cls(schema=schema, rows=rows, row_ids=row_ids)
        native_descriptors = _native.DescriptorSetVector()
        for descriptor_set in descriptors:
            if descriptor_set._native is None:
                raise TypeError("Cannot mix schema-backed and legacy descriptor sets.")
            native_descriptors.push_back(descriptor_set._native)
        return cls._from_native(_native._NativeDescriptorBatch.FromDescriptorSets(native_descriptors))

    @classmethod
    def from_molecules(
        cls,
        molecules: Iterable[Any],
        generator: Callable[..., DescriptorSet],
        /,
        **options: Any,
    ) -> DescriptorBatch:
        """Build a descriptor batch directly from molecules using a generator.

        :param molecules: Iterable of OpenEye molecules.
        :param generator: Callable mapping one molecule to one :class:`DescriptorSet`,
            e.g. :func:`morgan_descriptors`. Called as ``generator(mol, **options)``.
        :param options: Keyword options forwarded to ``generator`` for each molecule.
        :returns: A descriptor batch.
        :raises TypeError: When ``generator`` returns a non-:class:`DescriptorSet` value.
        """
        descriptors = []
        for mol in molecules:
            descriptor = generator(mol, **options)
            if not isinstance(descriptor, DescriptorSet):
                raise TypeError(
                    "DescriptorBatch.from_molecules generator must return DescriptorSet, "
                    f"got {type(descriptor).__name__}."
                )
            descriptors.append(descriptor)
        return cls.from_descriptors(descriptors)

    def __repr__(self) -> str:
        if self._native is None:
            columns = len(self._schema.definitions) if self._schema is not None else 0
            return f"DescriptorBatch(size={self.size}, columns={columns})"
        return f"DescriptorBatch(size={self.size}, value_type={self.value_type!r})"

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate schema-backed rows as ``{name: value}`` dictionaries.

        Each row contains every schema column, with ``None`` marking missing
        values. ``dict(batch)`` is intentionally not supported; use
        :meth:`to_dict` for a column-oriented mapping.

        :returns: Iterator over per-row descriptor dictionaries.
        """
        names = self.schema.names
        for row in self._schema_rows():
            yield {name: row[name] for name in names}

    def keys(self) -> NoReturn:
        """Refuse the mapping protocol so ``dict(batch)`` fails clearly.

        ``dict()`` checks for ``keys()`` before the iterable-of-pairs path, so
        this makes ``dict(batch)`` raise regardless of column count (a two-column
        batch would otherwise be misread as a single ``(key, value)`` pair). Use
        :meth:`to_dict` for column form or :meth:`to_records` for row form.

        :raises TypeError: Always, to reject ``dict(DescriptorBatch)``.
        """
        raise TypeError(
            "dict(DescriptorBatch) is unsupported; use to_dict() (columns) "
            "or to_records() (rows)."
        )

    def __getitem__(self, key: int | slice) -> dict[str, Any] | DescriptorBatch:
        """Return a single row dict or a sliced descriptor batch.

        :param key: Integer row index (negative indices allowed) or a slice.
        :returns: A ``{name: value}`` row dict for an integer key, or a new
            :class:`DescriptorBatch` for a slice key.
        """
        if isinstance(key, slice):
            rows = self._schema_rows()[key]
            return DescriptorBatch(schema=self.schema, rows=rows, row_ids=self._row_ids[key])
        names = self.schema.names
        row = self._schema_rows()[key]
        return {name: row[name] for name in names}

    def to_records(self) -> list[dict[str, Any]]:
        """Return all rows as a list of ``{name: value}`` dictionaries.

        :returns: List of per-row descriptor dictionaries.
        """
        return list(self)

    def to_list(self) -> list[dict[str, Any]]:
        """Return all rows as a list of ``{name: value}`` dictionaries.

        :returns: List of per-row descriptor dictionaries.
        """
        return list(self)

    def to_dict(self) -> dict[str, list[Any]]:
        """Return a column-oriented ``{name: [values...]}`` mapping.

        This is the counterpart to the row-oriented iteration protocol;
        ``dict(batch)`` is intentionally unsupported in favor of this method.

        :returns: Mapping of descriptor name to its column of values.
        """
        names = self.schema.names
        rows = self._schema_rows()
        return {name: [row[name] for row in rows] for name in names}

    def __add__(self, other: object) -> DescriptorBatch:
        """Vertically concatenate two schema-backed batches.

        :param other: Another :class:`DescriptorBatch` with an identical schema.
        :returns: A new batch containing the rows of both operands.
        :raises ValueError: When the two batches have different schemas.
        """
        if not isinstance(other, DescriptorBatch):
            return NotImplemented
        if self.schema.schema_id != other.schema.schema_id:
            raise ValueError("Cannot concatenate DescriptorBatches with different schemas.")
        rows = list(self._schema_rows()) + list(other._schema_rows())
        row_ids = tuple(self._row_ids) + tuple(other._row_ids)
        return DescriptorBatch(schema=self.schema, rows=rows, row_ids=row_ids)

    @property
    def schema(self) -> DescriptorSchema:
        """Descriptor schema for schema-backed batches."""
        return self._require_schema()

    @property
    def row_ids(self) -> tuple[str, ...]:
        """Schema-backed descriptor row identifiers."""
        self._require_schema()
        return self._row_ids

    def float_column(self, name: str) -> np.ndarray:
        """Return a float descriptor column.

        :param name: Descriptor name.
        :returns: NumPy float column.
        """
        self._require_column_kind(name, "float")
        return np.array(
            [
                np.nan if row[name] is None else row[name]
                for row in self._schema_rows()
            ],
            dtype=np.float64,
        )

    def int_column(self, name: str) -> np.ndarray:
        """Return an integer descriptor column.

        :param name: Descriptor name.
        :returns: NumPy integer column.
        """
        self._require_column_kind(name, "int")
        values = [row[name] for row in self._schema_rows()]
        dtype = object if any(value is None for value in values) else np.int64
        return np.array(values, dtype=dtype)

    def bool_column(self, name: str) -> np.ndarray:
        """Return a boolean descriptor column.

        :param name: Descriptor name.
        :returns: NumPy boolean column.
        """
        self._require_column_kind(name, "bool")
        values = [row[name] for row in self._schema_rows()]
        dtype = object if any(value is None for value in values) else np.bool_
        return np.array(values, dtype=dtype)

    def string_column(self, name: str) -> tuple[str | None, ...]:
        """Return a string descriptor column.

        :param name: Descriptor name.
        :returns: Tuple of string values.
        """
        self._require_column_kind(name, "string")
        return tuple(row[name] for row in self._schema_rows())

    def column_validity(self, name: str) -> np.ndarray:
        """Return a boolean mask indicating present descriptor values.

        :param name: Descriptor name.
        :returns: Boolean NumPy array where ``True`` marks present values.
        """
        self.schema.index(name)
        return np.array([row[name] is not None for row in self._schema_rows()], dtype=np.bool_)

    def subset(self, names: Sequence[str]) -> DescriptorBatch:
        """Return a schema-backed descriptor batch projected to named columns.

        :param names: Descriptor names to keep.
        :returns: Projected descriptor batch.
        """
        schema = self.schema.subset(names)
        rows = [{name: row[name] for name in schema.names} for row in self._schema_rows()]
        return DescriptorBatch(schema=schema, rows=rows, row_ids=self._row_ids)

    def to_arrow(self) -> Any:
        """Convert a schema-backed descriptor batch to a ``pyarrow.Table``.

        :returns: Arrow table with OEFP Python descriptor schema metadata.
        """
        import pyarrow as pa

        arrays = {
            definition.name: pa.array(
                [row[definition.name] for row in self._schema_rows()],
                type=_descriptor_arrow_type(pa, definition.value_type),
            )
            for definition in self.schema.definitions
        }
        table = pa.table(arrays)
        metadata = dict(table.schema.metadata or {})
        metadata[_DESCRIPTOR_SCHEMA_METADATA_KEY] = self.schema._metadata().encode("utf-8")
        metadata[_ROW_IDS_METADATA_KEY] = json.dumps(list(self._row_ids)).encode("utf-8")
        return table.replace_schema_metadata(metadata)

    @classmethod
    def from_arrow(cls, table: Any) -> DescriptorBatch:
        """Create a descriptor batch from a ``pyarrow.Table``.

        :param table: Arrow table produced by :meth:`to_arrow`.
        :returns: Schema-backed descriptor batch.
        :raises ValueError: When schema metadata is missing.
        """
        metadata = table.schema.metadata or {}
        if _DESCRIPTOR_SCHEMA_METADATA_KEY not in metadata:
            raise ValueError("Arrow table is missing OEFP descriptor schema metadata.")
        schema = DescriptorSchema._from_metadata(metadata[_DESCRIPTOR_SCHEMA_METADATA_KEY])
        has_row_ids = _ROW_IDS_METADATA_KEY in metadata
        row_ids: tuple[str, ...] = (
            tuple(json.loads(metadata[_ROW_IDS_METADATA_KEY].decode("utf-8")))
            if has_row_ids
            else ()
        )
        if len(schema.names) == 0:
            # A zero-column Arrow table always reports ``num_rows == 0``, so the
            # persisted row ids are the only authoritative row count available.
            # Validate that the physical table genuinely has zero descriptor columns.
            if table.num_columns != 0:
                raise ValueError(
                    "Arrow schema metadata indicates an empty schema but the physical "
                    "table has descriptor columns; the metadata may be stale or tampered."
                )
            row_count = len(row_ids)
        else:
            # For non-empty schemas the physical column length is authoritative.
            # Any row-id metadata must agree with it; a mismatch signals stale
            # metadata that would otherwise truncate rows or index out of range.
            row_count = int(table.num_rows)
            if has_row_ids and len(row_ids) != row_count:
                raise ValueError(
                    "Arrow row-id metadata length does not match the descriptor "
                    "table row count."
                )
            if not has_row_ids:
                row_ids = ("",) * row_count
        columns = {name: table.column(name).to_pylist() for name in schema.names}
        rows = [
            {name: columns[name][row_index] for name in schema.names}
            for row_index in range(row_count)
        ]
        return cls(schema=schema, rows=rows, row_ids=row_ids)

    def write_parquet(self, path: Any) -> None:
        """Write a schema-backed descriptor batch to a Parquet file.

        :param path: Destination file path.
        """
        import pyarrow.parquet as pq

        pq.write_table(self.to_arrow(), path)

    @classmethod
    def read_parquet(cls, path: Any) -> DescriptorBatch:
        """Read a schema-backed descriptor batch from a Parquet file.

        :param path: Source file path.
        :returns: Schema-backed descriptor batch.
        """
        import pyarrow.parquet as pq

        return cls.from_arrow(pq.read_table(path))

    def _schema_rows(self) -> tuple[dict[str, Any], ...]:
        if self._rows is None:
            raise TypeError("Legacy descriptor batches do not support named columns.")
        return self._rows

    def _require_native(self) -> Any:
        if self._native is None:
            raise TypeError(
                "Schema-backed descriptor batches do not expose legacy descriptor storage."
            )
        return self._native

    def _require_schema(self) -> DescriptorSchema:
        if self._schema is None:
            raise TypeError("Legacy descriptor batches do not expose a descriptor schema.")
        return self._schema

    def _require_column_kind(self, name: str, value_type: str) -> None:
        definition = self.schema.definitions[self.schema.index(name)]
        if definition.value_type != value_type:
            raise TypeError(
                f"Descriptor {name!r} has value type {definition.value_type!r}, "
                f"not {value_type!r}."
            )

    @property
    def value_type(self) -> str:
        """Shared descriptor key value type."""
        return _descriptor_value_type_name(self._require_native().ValueType())

    @property
    def size(self) -> int:
        """Number of descriptor rows."""
        if self._rows is not None:
            return len(self._rows)
        return int(self._require_native().Size())

    @property
    def entry_count(self) -> int:
        """Number of flattened descriptor entries."""
        return int(self._require_native().EntryCount())

    @property
    def string_keys(self) -> tuple[str, ...]:
        """Flattened string keys."""
        if self.value_type != "string":
            return ()
        return tuple(str(value) for value in self._require_native().StringKeys())

    @property
    def integer_keys(self) -> tuple[int, ...]:
        """Flattened integer keys."""
        if self.value_type != "integer":
            return ()
        keys = readonly_array_from_address(
            self,
            self._require_native().IntegerKeyDataAddress(),
            (self._require_native().EntryCount(),),
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
            self._require_native().FloatKeyDataAddress(),
            (self._require_native().EntryCount(),),
            np.dtype(np.float64),
        )
        return tuple(float(value) for value in keys)

    @property
    def counts(self) -> np.ndarray:
        """Read-only view of flattened descriptor counts."""
        return readonly_array_from_address(
            self,
            self._require_native().CountDataAddress(),
            (self._require_native().EntryCount(),),
            np.dtype(np.uint32),
        )

    @property
    def offsets(self) -> np.ndarray:
        """Read-only view of row offsets into flattened keys and counts."""
        return readonly_array_from_address(
            self,
            self._require_native().RowOffsetDataAddress(),
            (self._require_native().Size() + 1,),
            np.dtype(np.uint64),
        )

    @property
    def spec(self) -> DescriptorSpec:
        """Read-only descriptor metadata for all rows."""
        return _descriptor_spec(self._require_native().Spec())


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

    @classmethod
    def from_molecules(
        cls,
        molecules: Iterable[Any],
        generator: Callable[..., OEFP],
        /,
        **options: Any,
    ) -> OEFPBatch:
        """Build a batch directly from molecules using a fingerprint generator.

        :param molecules: Iterable of OpenEye molecules.
        :param generator: Callable mapping one molecule to one :class:`OEFP`,
            e.g. :func:`morgan_fingerprint`. Called as ``generator(mol, **options)``.
        :param options: Keyword options forwarded to ``generator`` for each molecule.
        :returns: A dense binary batch.
        :raises TypeError: When ``generator`` returns a non-:class:`OEFP` value.
        """
        fingerprints = []
        for mol in molecules:
            fingerprint = generator(mol, **options)
            if not isinstance(fingerprint, OEFP):
                raise TypeError(
                    "OEFPBatch.from_molecules generator must return OEFP, "
                    f"got {type(fingerprint).__name__}."
                )
            fingerprints.append(fingerprint)
        return cls.from_fingerprints(fingerprints)

    def __repr__(self) -> str:
        return (
            f"OEFPBatch(size={self.size}, num_bits={self.num_bits}, "
            f"source_type={self.spec.source_type!r})"
        )

    def __len__(self) -> int:
        return self.size

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

    @classmethod
    def from_molecules(
        cls,
        molecules: Iterable[Any],
        generator: Callable[..., OEFPCount],
        /,
        **options: Any,
    ) -> OEFPCountBatch:
        """Build a counted batch directly from molecules using a generator.

        :param molecules: Iterable of OpenEye molecules.
        :param generator: Callable mapping one molecule to one :class:`OEFPCount`,
            e.g. :func:`morgan_count_fingerprint`. Called as ``generator(mol, **options)``.
        :param options: Keyword options forwarded to ``generator`` for each molecule.
        :returns: A counted batch.
        :raises TypeError: When ``generator`` returns a non-:class:`OEFPCount` value.
        """
        fingerprints = []
        for mol in molecules:
            fingerprint = generator(mol, **options)
            if not isinstance(fingerprint, OEFPCount):
                raise TypeError(
                    "OEFPCountBatch.from_molecules generator must return OEFPCount, "
                    f"got {type(fingerprint).__name__}."
                )
            fingerprints.append(fingerprint)
        return cls.from_fingerprints(fingerprints)

    def __repr__(self) -> str:
        # num_bits is the (possibly unfolded) identifier domain rather than a fold
        # size, so entry_count is the relevant occupancy data.
        return (
            f"OEFPCountBatch(size={self.size}, entry_count={self.entry_count}, "
            f"source_type={self.spec.source_type!r})"
        )

    def __len__(self) -> int:
        return self.size

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

    @classmethod
    def from_molecules(
        cls,
        molecules: Iterable[Any],
        generator: Callable[..., OEFPSparse],
        /,
        **options: Any,
    ) -> OEFPSparseBatch:
        """Build a sparse binary batch directly from molecules using a generator.

        :param molecules: Iterable of OpenEye molecules.
        :param generator: Callable mapping one molecule to one :class:`OEFPSparse`,
            e.g. :func:`morgan_sparse_fingerprint`. Called as ``generator(mol, **options)``.
        :param options: Keyword options forwarded to ``generator`` for each molecule.
        :returns: A sparse binary batch.
        :raises TypeError: When ``generator`` returns a non-:class:`OEFPSparse` value.
        """
        fingerprints = []
        for mol in molecules:
            fingerprint = generator(mol, **options)
            if not isinstance(fingerprint, OEFPSparse):
                raise TypeError(
                    "OEFPSparseBatch.from_molecules generator must return OEFPSparse, "
                    f"got {type(fingerprint).__name__}."
                )
            fingerprints.append(fingerprint)
        return cls.from_fingerprints(fingerprints)

    def __repr__(self) -> str:
        # num_bits is the unfolded identifier domain rather than a fold size, so
        # entry_count is the relevant occupancy data.
        return (
            f"OEFPSparseBatch(size={self.size}, entry_count={self.entry_count}, "
            f"source_type={self.spec.source_type!r})"
        )

    def __len__(self) -> int:
        return self.size

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

    def __repr__(self) -> str:
        if self.name == "tversky":
            return (
                f"Metric(name={self.name!r}, type={self.type!r}, "
                f"alpha={self.alpha}, beta={self.beta})"
            )
        return f"Metric(name={self.name!r}, type={self.type!r})"

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
        use_features: bool = False,
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
            use_features,
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

    def __repr__(self) -> str:
        options = self._native.Options()
        return f"MorganGenerator(radius={options.radius}, num_bits={options.num_bits})"


class TopologicalAtomPairGenerator:
    """Reusable generator for folded binary topological Atom Pair fingerprints."""

    def __init__(
        self,
        *,
        min_distance: int = 1,
        max_distance: int = 30,
        num_bits: int = 2048,
        use_chirality: bool = False,
        count_simulation: bool = True,
        count_bounds: Sequence[int] | None = None,
    ) -> None:
        options = _atom_pair_options(
            min_distance,
            max_distance,
            num_bits,
            use_chirality,
            True,
            count_simulation,
            count_bounds,
        )
        self._native = _native._NativeAtomPairGenerator(options)

    def fingerprint(self, mol: Any) -> OEFP:
        """Generate a folded dense binary topological Atom Pair fingerprint."""
        return OEFP._from_native(self._native.Fingerprint(mol))

    def __repr__(self) -> str:
        options = self._native.Options()
        return (
            f"{type(self).__name__}(min_distance={options.min_distance}, "
            f"max_distance={options.max_distance}, num_bits={options.num_bits})"
        )


class TopologicalTorsionsGenerator:
    """Reusable generator for folded binary Topological Torsions fingerprints."""

    def __init__(
        self,
        *,
        torsion_atom_count: int = 4,
        num_bits: int = 2048,
        use_chirality: bool = False,
        count_simulation: bool = True,
        count_bounds: Sequence[int] | None = None,
    ) -> None:
        options = _topological_torsions_options(
            torsion_atom_count,
            num_bits,
            use_chirality,
            count_simulation,
            count_bounds,
        )
        self._native = _native._NativeTopologicalTorsionsGenerator(options)

    def fingerprint(self, mol: Any) -> OEFP:
        """Generate a folded dense binary Topological Torsions fingerprint."""
        return OEFP._from_native(self._native.Fingerprint(mol))

    def __repr__(self) -> str:
        options = self._native.Options()
        return (
            f"TopologicalTorsionsGenerator(torsion_atom_count={options.torsion_atom_count}, "
            f"num_bits={options.num_bits})"
        )


class AtomPairGenerator(TopologicalAtomPairGenerator):
    """Compatibility generator for RDKit-style Atom Pair options.

    ``use_2d=True`` is the topological/connectivity-distance model. Passing
    ``use_2d=False`` selects the separate Distance Atom Pair model, which is
    not implemented.
    """

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
        if _is_schema_descriptor_set(a) or _is_schema_descriptor_set(b):
            _raise_schema_descriptor_compare_error()
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
        if _is_schema_descriptor_set(a) or _is_schema_descriptor_batch(b):
            _raise_schema_descriptor_compare_error()
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
        if _is_schema_descriptor_batch(a) or _is_schema_descriptor_batch(b):
            _raise_schema_descriptor_compare_error()
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
        if _is_schema_descriptor_batch(batch):
            _raise_schema_descriptor_compare_error()
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
    use_features: bool = False,
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
        use_features_bool,
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
        use_features,
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
    options.use_features = use_features_bool
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
    use_features: bool,
    use_bond_types: bool,
    only_nonzero_invariants: bool,
    include_ring_membership: bool,
    include_redundant_environments: bool,
    count_simulation: bool,
    count_bounds: Sequence[int] | None,
) -> tuple[int, int, bool, bool, bool, bool, bool, bool, bool, tuple[int, ...]]:
    radius_int = _uint32_option("Morgan", "radius", radius, positive=False)
    num_bits_int = _uint32_option("Morgan", "num_bits", num_bits, positive=True)
    normalized_count_bounds = _normalized_count_bounds("Morgan", count_bounds)
    if count_simulation and not normalized_count_bounds:
        raise ValueError("Morgan count_bounds cannot be empty when count simulation is enabled.")
    if count_simulation and len(normalized_count_bounds) >= num_bits_int:
        raise ValueError("Morgan count_bounds length must be smaller than num_bits.")
    return (
        radius_int,
        num_bits_int,
        bool(use_chirality),
        bool(use_features),
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
    use_features: bool,
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
        use_features=use_features,
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


@lru_cache(maxsize=32)
def _cached_topological_torsions_generator(
    torsion_atom_count: int,
    num_bits: int,
    use_chirality: bool,
    count_simulation: bool,
    count_bounds: tuple[int, ...],
) -> TopologicalTorsionsGenerator:
    return TopologicalTorsionsGenerator(
        torsion_atom_count=torsion_atom_count,
        num_bits=num_bits,
        use_chirality=use_chirality,
        count_simulation=count_simulation,
        count_bounds=count_bounds,
    )


def morgan_fingerprint(
    mol: Any,
    *,
    radius: int = 2,
    num_bits: int = 2048,
    use_chirality: bool = False,
    use_features: bool = False,
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
            use_features,
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
    if not use_2d:
        raise ValueError(
            "Distance Atom Pair requires existing 3D coordinates and is not implemented yet."
        )

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
    native = _native.MakeAtomPairDescriptors(mol, options)
    return _legacy_counted_string_descriptor(native, _descriptor_spec(native.Spec()))


def topological_atom_pair_fingerprint(
    mol: Any,
    *,
    min_distance: int = 1,
    max_distance: int = 30,
    num_bits: int = 2048,
    use_chirality: bool = False,
    count_simulation: bool = True,
    count_bounds: Sequence[int] | None = None,
) -> OEFP:
    """Generate a folded binary topological Atom Pair fingerprint."""
    return atom_pair_fingerprint(
        mol,
        min_distance=min_distance,
        max_distance=max_distance,
        num_bits=num_bits,
        use_chirality=use_chirality,
        use_2d=True,
        count_simulation=count_simulation,
        count_bounds=count_bounds,
    )


def topological_atom_pair_count_fingerprint(
    mol: Any,
    *,
    min_distance: int = 1,
    max_distance: int = 30,
    num_bits: int = 2048,
    use_chirality: bool = False,
) -> OEFPCount:
    """Generate a folded count topological Atom Pair fingerprint."""
    return atom_pair_count_fingerprint(
        mol,
        min_distance=min_distance,
        max_distance=max_distance,
        num_bits=num_bits,
        use_chirality=use_chirality,
        use_2d=True,
    )


def topological_atom_pair_sparse_fingerprint(
    mol: Any,
    *,
    min_distance: int = 1,
    max_distance: int = 30,
    use_chirality: bool = False,
    count_simulation: bool = True,
    count_bounds: Sequence[int] | None = None,
) -> OEFPSparse:
    """Generate a sparse binary topological Atom Pair fingerprint."""
    return atom_pair_sparse_fingerprint(
        mol,
        min_distance=min_distance,
        max_distance=max_distance,
        use_chirality=use_chirality,
        use_2d=True,
        count_simulation=count_simulation,
        count_bounds=count_bounds,
    )


def topological_atom_pair_sparse_count_fingerprint(
    mol: Any,
    *,
    min_distance: int = 1,
    max_distance: int = 30,
    use_chirality: bool = False,
) -> OEFPCount:
    """Generate a sparse count topological Atom Pair fingerprint."""
    return atom_pair_sparse_count_fingerprint(
        mol,
        min_distance=min_distance,
        max_distance=max_distance,
        use_chirality=use_chirality,
        use_2d=True,
    )


def topological_atom_pair_descriptors(
    mol: Any,
    *,
    min_distance: int = 1,
    max_distance: int = 30,
    use_chirality: bool = False,
) -> DescriptorSet:
    """Generate raw topological Atom Pair descriptors as counted string keys."""
    return atom_pair_descriptors(
        mol,
        min_distance=min_distance,
        max_distance=max_distance,
        use_chirality=use_chirality,
        use_2d=True,
    )


def topological_torsions_fingerprint(
    mol: Any,
    *,
    torsion_atom_count: int = 4,
    num_bits: int = 2048,
    use_chirality: bool = False,
    count_simulation: bool = True,
    count_bounds: Sequence[int] | None = None,
) -> OEFP:
    """Generate an RDKit-compatible folded binary Topological Torsions fingerprint."""
    generator = _cached_topological_torsions_generator(
        *_normalized_topological_torsions_values(
            torsion_atom_count,
            num_bits,
            use_chirality,
            count_simulation,
            count_bounds,
        )
    )
    return generator.fingerprint(mol)


def topological_torsions_count_fingerprint(
    mol: Any,
    *,
    torsion_atom_count: int = 4,
    num_bits: int = 2048,
    use_chirality: bool = False,
) -> OEFPCount:
    """Generate an RDKit-compatible folded count Topological Torsions fingerprint."""
    options = _topological_torsions_options(
        torsion_atom_count,
        num_bits,
        use_chirality,
        False,
        None,
    )
    return OEFPCount._from_native(_native.MakeTopologicalTorsionsCountFingerprint(mol, options))


def topological_torsions_sparse_fingerprint(
    mol: Any,
    *,
    torsion_atom_count: int = 4,
    use_chirality: bool = False,
    count_simulation: bool = True,
    count_bounds: Sequence[int] | None = None,
) -> OEFPSparse:
    """Generate an RDKit-compatible sparse binary Topological Torsions fingerprint."""
    options = _topological_torsions_options(
        torsion_atom_count,
        2048,
        use_chirality,
        count_simulation,
        count_bounds,
    )
    return OEFPSparse._from_native(_native.MakeTopologicalTorsionsSparseFingerprint(mol, options))


def topological_torsions_sparse_count_fingerprint(
    mol: Any,
    *,
    torsion_atom_count: int = 4,
    use_chirality: bool = False,
) -> OEFPCount64:
    """Generate an RDKit-compatible raw sparse-count Topological Torsions fingerprint."""
    options = _topological_torsions_options(
        torsion_atom_count,
        2048,
        use_chirality,
        False,
        None,
    )
    return OEFPCount64._from_native(
        _native.MakeTopologicalTorsionsSparseCountFingerprint(mol, options)
    )


def topological_torsions_descriptors(
    mol: Any,
    *,
    torsion_atom_count: int = 4,
    use_chirality: bool = False,
) -> DescriptorSet:
    """Generate raw Topological Torsions descriptors as counted string keys."""
    options = _topological_torsions_options(
        torsion_atom_count,
        2048,
        use_chirality,
        False,
        None,
    )
    native = _native.MakeTopologicalTorsionsDescriptors(mol, options)
    return _legacy_counted_string_descriptor(native, _descriptor_spec(native.Spec()))


def distance_atom_pair_fingerprint(mol: Any, **_: Any) -> OEFP:
    """Reject unsupported 3D coordinate-distance Atom Pair fingerprints."""
    _require_distance_atom_pair_3d(mol)
    _raise_distance_atom_pair_not_implemented()


def distance_atom_pair_count_fingerprint(mol: Any, **_: Any) -> OEFPCount:
    """Reject unsupported 3D coordinate-distance Atom Pair count fingerprints."""
    _require_distance_atom_pair_3d(mol)
    _raise_distance_atom_pair_not_implemented()


def distance_atom_pair_sparse_fingerprint(mol: Any, **_: Any) -> OEFPSparse:
    """Reject unsupported 3D coordinate-distance Atom Pair sparse fingerprints."""
    _require_distance_atom_pair_3d(mol)
    _raise_distance_atom_pair_not_implemented()


def distance_atom_pair_sparse_count_fingerprint(mol: Any, **_: Any) -> OEFPCount:
    """Reject unsupported 3D coordinate-distance Atom Pair sparse count fingerprints."""
    _require_distance_atom_pair_3d(mol)
    _raise_distance_atom_pair_not_implemented()


def distance_atom_pair_descriptors(mol: Any, **_: Any) -> DescriptorSet:
    """Reject unsupported 3D coordinate-distance Atom Pair descriptors."""
    _require_distance_atom_pair_3d(mol)
    _raise_distance_atom_pair_not_implemented()


def morgan_descriptors(
    mol: Any,
    *,
    radius: int = 2,
    use_chirality: bool = False,
    use_features: bool = False,
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
        use_features,
        use_bond_types,
        only_nonzero_invariants,
        include_ring_membership,
        include_redundant_environments,
    )
    native = _native.MakeMorganDescriptors(mol, options)
    return _legacy_counted_integer_descriptor(native, _descriptor_spec(native.Spec()))


def mordred_descriptors(mol: Any) -> DescriptorSet:
    """Generate schema-backed Mordred-compatible descriptors.

    :param mol: OpenEye molecule to describe.
    :returns: Descriptor row backed by :func:`mordred_schema`.
    """
    native = _native.MakeMordredDescriptors(mol)
    schema = mordred_schema()
    values = {
        definition.name: _mordred_value_from_native(native, definition)
        for definition in schema.definitions
    }
    return DescriptorSet(schema, values)


def _descriptor_definition_from_native(native_definition: Any) -> DescriptorDefinition:
    """Build a Python descriptor definition from a native schema definition."""
    return DescriptorDefinition(
        str(native_definition.name),
        _descriptor_kind_name(native_definition.value_kind),
        group=str(native_definition.group),
        source_name=str(native_definition.source_name),
        source_type=str(native_definition.source_type),
        source_version=str(native_definition.source_version),
        parameters=str(native_definition.parameters),
        units=str(native_definition.units),
        description=str(native_definition.description),
        prerequisites=int(native_definition.prerequisites),
        canonical_id=str(native_definition.canonical_id),
    )


def _schema_from_native(native_schema: Any) -> DescriptorSchema:
    """Build a Python descriptor schema from a native merged schema.

    Uses :meth:`DescriptorSchema._allow_empty` so a calculator that resolves to
    zero columns still yields a valid (empty) schema.
    """
    definitions = [
        _descriptor_definition_from_native(native_schema.Definition(index))
        for index in range(native_schema.Size())
    ]
    return DescriptorSchema._allow_empty(definitions)


def _batch_column_values(native_batch: Any, definition: DescriptorDefinition) -> list[Any]:
    """Extract one column of Python values, placing ``None`` for missing entries."""
    validity = list(native_batch.ColumnValidity(definition.name))
    raw: list[Any]
    if definition.value_type == "bool":
        raw = [bool(value) for value in native_batch.BoolColumn(definition.name)]
    elif definition.value_type == "int":
        raw = [int(value) for value in native_batch.IntColumn(definition.name)]
    elif definition.value_type == "float":
        raw = [float(value) for value in native_batch.FloatColumn(definition.name)]
    elif definition.value_type == "string":
        raw = list(native_batch.StringColumn(definition.name))
    else:
        raise ValueError(f"Unsupported descriptor value type: {definition.value_type!r}.")
    return [value if present else None for value, present in zip(raw, validity, strict=True)]


class MordredDescriptorSource:
    """Descriptor source for the curated Mordred descriptor family."""

    def __init__(self) -> None:
        self._native = _native.MordredDescriptorSource()


class OpenEyePropertyDescriptorSource:
    """Descriptor source for OpenEye molecular-property descriptors."""

    def __init__(self) -> None:
        self._native = _native.OpenEyePropertyDescriptorSource()


DescriptorSource = MordredDescriptorSource | OpenEyePropertyDescriptorSource


class DescriptorCalculator:
    """Compute merged, deduplicated descriptors from one or more sources.

    Sources are resolved in order; when two sources expose the same curated
    ``canonical_id`` the first source wins and later duplicates are dropped.

    :param sources: Sequence whose items are either a bare descriptor source
        wrapper or a ``(source, names)`` tuple, where ``names`` is a sequence of
        descriptor names to keep from that source (or ``None`` to keep all).
    """

    def __init__(
        self,
        sources: Sequence[
            DescriptorSource | tuple[DescriptorSource, Sequence[str] | None]
        ],
    ) -> None:
        entries = _native.DescriptorSourceEntryVector()
        for item in sources:
            if isinstance(item, tuple):
                source, names = item
            else:
                source, names = item, None
            if names is None:
                entry = _native.DescriptorSourceEntry(source._native)
            else:
                selection = _native.DescriptorSelection.Names(
                    _native.StringVector(list(names))
                )
                entry = _native.DescriptorSourceEntry(source._native, selection)
            entries.push_back(entry)
        self._native = _native._NativeDescriptorCalculator(entries)

    @property
    def schema(self) -> DescriptorSchema:
        """Merged, deduplicated descriptor schema for all resolved columns."""
        return _schema_from_native(self._native.Schema())

    def compute(self, mol: Any) -> DescriptorSet:
        """Compute one schema-backed descriptor row for a molecule.

        :param mol: OpenEye molecule to describe.
        :returns: Descriptor row backed by :attr:`schema`.
        """
        schema = self.schema
        native = self._native.Compute(mol)
        values = {
            definition.name: _mordred_value_from_native(native, definition)
            for definition in schema.definitions
        }
        return DescriptorSet(schema, values)

    def calculate_batch(self, molecules: Iterable[Any]) -> DescriptorBatch:
        """Compute a schema-backed descriptor batch for many molecules.

        :param molecules: Iterable of OpenEye molecules.
        :returns: Descriptor batch backed by :attr:`schema`.
        """
        schema = self.schema
        native = self._native.CalculateBatch(list(molecules))
        row_ids = list(native.RowIds())
        row_count = int(native.Size())
        if not schema.definitions:
            rows: list[dict[str, Any]] = [{} for _ in range(row_count)]
            return DescriptorBatch(schema=schema, rows=rows, row_ids=row_ids)
        columns = {
            definition.name: _batch_column_values(native, definition)
            for definition in schema.definitions
        }
        rows = [
            {name: columns[name][row_index] for name in schema.names}
            for row_index in range(row_count)
        ]
        return DescriptorBatch(schema=schema, rows=rows, row_ids=row_ids)


def morgan_fingerprint_with_mapping(
    mol: Any,
    *,
    radius: int = 2,
    num_bits: int = 2048,
    use_chirality: bool = False,
    use_features: bool = False,
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
        use_features,
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
    use_features: bool = False,
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
        use_features,
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
    use_features: bool = False,
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
        use_features,
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
    use_features: bool = False,
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
        use_features,
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
    use_features: bool = False,
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
        use_features,
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
    use_features: bool = False,
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
        use_features,
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
    use_features: bool = False,
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
        use_features,
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
