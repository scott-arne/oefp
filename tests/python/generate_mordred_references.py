"""Generate Mordred descriptor schema and reference fixtures.

This script uses the local Mordred checkout with compatibility shims for the
newer NumPy and NetworkX versions in the default development environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any

DEFAULT_MORDRED_SOURCE = Path("/Users/johnss51/Development/python/mordred")
EXPECTED_MORDRED_VERSION = "1.2.0"
SOURCE_VERSION = f"Mordred-{EXPECTED_MORDRED_VERSION}"
DESCRIPTOR_PREREQUISITE_NONE = 0
DESCRIPTOR_PREREQUISITE_COORDINATES_3D = 1 << 2

SMILES_PANEL = [
    "CCO",
    "c1ccncc1",
    "CC(C)(C)Cl",
    "O=S(=O)(N)C1=CC=CC=C1",
    "CI",
    "C[Na]",
    "C[N+](C)(C)CC(=O)[O-]",
    "C1=CC2=C(C=C1)C=CC=C2",
    "C12C3C4C1C5C2C3C45",
    "C1C2CC3CC1CC(C2)C3",
    "CC#N",
    "FC(F)(F)c1ccc(Br)cc1",
    "CN",
    "c1ccccc1[N+](=O)[O-]",
    "CC(=O)O",
    "OP(=O)(O)O",
    "C",
    "CC",
    "CCC",
    "CCCCCC",
    "C1CCCCC1",
    "C1CC1C",
    "CCCCCCCCCCCCCCCC",
    "O=[Se]=O",
    "[13CH4]",
    "COC(=O)c1ccc(OCC)c(O)c1C(=O)OCC",
]


def _apply_compatibility_shims() -> None:
    """Apply compatibility patches required by the local Mordred checkout."""
    import numpy as np

    setattr(np, "float", float)
    setattr(np, "int", int)
    setattr(np, "bool", np.bool_)
    setattr(np, "object", object)
    setattr(np, "product", np.prod)

    original_sum = np.sum

    def compatible_sum(value: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(value, types.GeneratorType):
            value = list(value)
        return original_sum(value, *args, **kwargs)

    np.sum = compatible_sum  # type: ignore[assignment]

    import networkx as nx  # type: ignore[import-untyped]

    if not hasattr(nx, "biconnected_component_subgraphs"):

        def biconnected_component_subgraphs(graph: Any, copy: bool = True) -> Any:
            for component in nx.biconnected_components(graph):
                subgraph = graph.subgraph(component)
                yield subgraph.copy() if copy else subgraph

        nx.biconnected_component_subgraphs = biconnected_component_subgraphs


def _descriptor_group(descriptor: Any) -> str:
    module = descriptor.__class__.__module__.rsplit(".", maxsplit=1)[-1]
    return f"mordred:{module}"


def _value_kind(descriptor: Any) -> str:
    rtype = getattr(descriptor, "rtype", float)
    if rtype is bool:
        return "bool"
    if rtype is int:
        return "int"
    if rtype is float:
        return "float"
    return "string"


def _descriptor_parameters(descriptor: Any) -> str:
    payload = descriptor.to_json()
    args = payload.get("args", {})
    return json.dumps(args, sort_keys=True, separators=(",", ":")) if args else ""


def _descriptor_description(descriptor: Any) -> str:
    description = getattr(descriptor, "description", "")
    if callable(description):
        try:
            return str(description())
        except Exception as exc:
            raise RuntimeError(f"Could not extract Mordred description for {descriptor}.") from exc
    return str(description)


def _json_value(value: Any) -> Any:
    import numpy as np
    from mordred.error import Error, Missing, MissingValueBase  # type: ignore[import-not-found]

    if isinstance(value, MissingValueBase):
        if isinstance(value, Missing):
            state = "missing"
        elif isinstance(value, Error):
            state = "error"
        else:
            state = "missing"
        return {"error_type": type(value).__name__, "state": state}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return {"error_type": str(number), "state": "nonfinite"}
        return number
    return str(value)


def _load_mordred(mordred_source: Path) -> Any:
    resolved_source = mordred_source.expanduser().resolve()
    if not resolved_source.exists():
        raise FileNotFoundError(f"Mordred source path does not exist: {resolved_source}")

    sys.path.insert(0, str(resolved_source))
    _apply_compatibility_shims()

    import mordred  # type: ignore[import-not-found]
    from mordred import Calculator, descriptors  # type: ignore[import-not-found]

    actual_version = getattr(mordred, "__version__", None)
    if actual_version != EXPECTED_MORDRED_VERSION:
        raise RuntimeError(
            f"Expected Mordred {EXPECTED_MORDRED_VERSION}, imported {actual_version!r}."
        )
    imported_path = Path(mordred.__file__).resolve()
    if not imported_path.is_relative_to(resolved_source):
        raise RuntimeError(
            f"Imported Mordred from {imported_path}, outside requested source {resolved_source}."
        )

    return Calculator(descriptors, ignore_3D=False)


def _schema_id(definitions: list[dict[str, Any]]) -> str:
    def append_field(text: str, value: str) -> str:
        return f"{text}{len(value.encode('utf-8'))}:{value}"

    serialized = "oefp-descriptor-schema-v1\n"
    for definition in definitions:
        serialized = append_field(serialized, definition["name"])
        serialized += "|"
        serialized += definition["value_kind"]
        serialized += "|"
        serialized = append_field(serialized, definition["group"])
        serialized += "|"
        serialized = append_field(serialized, definition["source_name"])
        serialized += "|"
        serialized = append_field(serialized, definition["source_type"])
        serialized += "|"
        serialized = append_field(serialized, definition["source_version"])
        serialized += "|"
        serialized = append_field(serialized, definition["parameters"])
        serialized += "|"
        serialized = append_field(serialized, definition.get("units", ""))
        serialized += "|"
        serialized = append_field(serialized, definition["description"])
        serialized += "|"
        serialized += str(int(definition.get("prerequisites", DESCRIPTOR_PREREQUISITE_NONE)))
        serialized += "|-\n"

    value = 14695981039346656037
    for byte in serialized.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def _reference_payload(mordred_source: Path, descriptor_source: str) -> dict[str, Any]:
    from rdkit import Chem

    calculator = _load_mordred(mordred_source)
    definitions = [
        {
            "name": str(descriptor),
            "group": _descriptor_group(descriptor),
            "value_kind": _value_kind(descriptor),
            "source_name": "Mordred",
            "source_type": descriptor.__class__.__module__.rsplit(".", maxsplit=1)[-1],
            "source_version": SOURCE_VERSION,
            "parameters": _descriptor_parameters(descriptor),
            "units": "",
            "description": _descriptor_description(descriptor),
            "prerequisites": (
                DESCRIPTOR_PREREQUISITE_COORDINATES_3D
                if getattr(descriptor, "require_3D", False)
                else DESCRIPTOR_PREREQUISITE_NONE
            ),
        }
        for descriptor in calculator.descriptors
    ]

    rows = []
    for smiles in SMILES_PANEL:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
        values = [_json_value(value) for value in calculator(mol)]
        rows.append({"smiles": smiles, "values": values})

    return {
        "schema_id": _schema_id(definitions),
        "source": {
            "descriptor_source": descriptor_source,
            "name": "Mordred",
            "version": SOURCE_VERSION,
            "ignore_3D": False,
        },
        "smiles_panel": SMILES_PANEL,
        "definitions": definitions,
        "reference_rows": rows,
    }


def _cpp_kind(value_kind: str) -> str:
    return {
        "bool": "DescriptorValueKind::Bool",
        "int": "DescriptorValueKind::Int",
        "float": "DescriptorValueKind::Float",
        "string": "DescriptorValueKind::String",
    }[value_kind]


def _cpp_string(value: str) -> str:
    return json.dumps(value)


def _write_cpp_schema(payload: dict[str, Any], output: Path) -> None:
    definitions = payload["definitions"]
    lines = [
        '#include "oefp/mordred.h"',
        "",
        "#include <array>",
        "#include <memory>",
        "#include <string>",
        "#include <utility>",
        "#include <vector>",
        "",
        "namespace OEFP {",
        "namespace {",
        "",
        "struct StaticMordredDefinition {",
        "    const char* name;",
        "    DescriptorValueKind value_kind;",
        "    const char* group;",
        "    const char* source_type;",
        "    const char* parameters;",
        "    const char* description;",
        "    DescriptorPrerequisites prerequisites;",
        "};",
        "",
        f"constexpr std::array<StaticMordredDefinition, {len(definitions)}> kMordredDefinitions{{{{",
    ]
    for definition in definitions:
        lines.append(
            "    {"
            f"{_cpp_string(definition['name'])}, "
            f"{_cpp_kind(definition['value_kind'])}, "
            f"{_cpp_string(definition['group'])}, "
            f"{_cpp_string(definition['source_type'])}, "
            f"{_cpp_string(definition['parameters'])}, "
            f"{_cpp_string(definition['description'])}, "
            f"{int(definition.get('prerequisites', DESCRIPTOR_PREREQUISITE_NONE))}u"
            "},"
        )
    lines.extend(
        [
            "}};",
            "",
            "} // namespace",
            "",
            "std::shared_ptr<const DescriptorSchema> MordredDescriptorSchema() {",
            "    static const std::shared_ptr<const DescriptorSchema> schema = [] {",
            "        std::vector<DescriptorDefinition> definitions;",
            "        definitions.reserve(kMordredDefinitions.size());",
            "        for (const auto& item : kMordredDefinitions) {",
            "            DescriptorDefinition definition;",
            "            definition.name = item.name;",
            "            definition.value_kind = item.value_kind;",
            "            definition.group = item.group;",
            "            definition.source_name = \"Mordred\";",
            "            definition.source_type = item.source_type;",
            f"            definition.source_version = \"{SOURCE_VERSION}\";",
            "            definition.parameters = item.parameters;",
            "            definition.description = item.description;",
            "            definition.prerequisites = item.prerequisites;",
            "            definitions.push_back(std::move(definition));",
            "        }",
            "        return std::make_shared<const DescriptorSchema>(std::move(definitions));",
            "    }();",
            "    return schema;",
            "}",
            "",
            "} // namespace OEFP",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mordred-source",
        type=Path,
        default=Path(os.environ.get("OEFP_MORDRED_SOURCE", DEFAULT_MORDRED_SOURCE)),
        help="Local Mordred source checkout to import.",
    )
    parser.add_argument(
        "--descriptor-source",
        default=os.environ.get("OEFP_MORDRED_SOURCE_ID", "local-mordred-1.2.0"),
        help="Stable source identifier recorded in the fixture.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpp-output", type=Path)
    args = parser.parse_args()

    payload = _reference_payload(args.mordred_source, args.descriptor_source)
    if len(payload["definitions"]) != 1826:
        raise RuntimeError(f"Expected 1826 Mordred definitions, got {len(payload['definitions'])}.")
    for row in payload["reference_rows"]:
        if len(row["values"]) != 1826:
            raise RuntimeError(f"Expected 1826 values for {row['smiles']}.")

    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.cpp_output is not None:
        _write_cpp_schema(payload, args.cpp_output)


if __name__ == "__main__":
    main()
