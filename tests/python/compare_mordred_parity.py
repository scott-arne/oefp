"""Compare OEFP Mordred descriptors against the committed static fixture."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REFERENCE_FIXTURE = Path(__file__).with_name("mordred_references.json")
DEFAULT_POLICY = Path(__file__).with_name("mordred_divergences.json")
EXACT_REL_TOLERANCE = 1e-8
EXACT_ABS_TOLERANCE = 1e-8


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


def _definition_names(payload: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(definition["name"]) for definition in payload["definitions"])


def _definitions_by_name(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(definition["name"]): definition
        for definition in payload["definitions"]
    }


def _selected_names(
    payload: Mapping[str, Any],
    families: Sequence[str],
    names: Sequence[str],
    include_all: bool,
) -> tuple[str, ...]:
    definitions = payload["definitions"]
    known_names = set(_definition_names(payload))
    selected: list[str] = []

    if include_all:
        selected.extend(_definition_names(payload))

    if families:
        family_set = set(families)
        selected.extend(
            str(definition["name"])
            for definition in definitions
            if definition.get("source_type") in family_set
        )

    for name in names:
        if name not in known_names:
            raise ValueError(f"Unknown descriptor name: {name}")
        selected.append(name)

    if not selected:
        raise ValueError("Select descriptors with --families, --names, or --all.")

    return tuple(dict.fromkeys(selected))


def _descriptor_policy(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    policies = payload.get("policies", [])
    descriptor_counts = Counter(str(policy["descriptor"]) for policy in policies)
    duplicate_descriptors = sorted(
        descriptor
        for descriptor, count in descriptor_counts.items()
        if count > 1
    )
    if duplicate_descriptors:
        duplicates = ", ".join(duplicate_descriptors)
        raise ValueError(f"Duplicate descriptor policy entries: {duplicates}")

    return {str(policy["descriptor"]): policy for policy in policies}


def _row_divergence_policy(
    payload: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(row["descriptor"]), str(row["smiles"])): row
        for row in payload.get("row_divergences", [])
        if row.get("status") == "openeye_divergent"
    }


def _reference_values_by_name(
    descriptor_names: Sequence[str],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return dict(zip(descriptor_names, row["values"], strict=True))


def _reference_value(value: Any) -> Any:
    if isinstance(value, dict):
        return None
    return value


def _is_absent_reference_value(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("state") in {"missing", "error", "nonfinite"}
    )


def _values_match(reference: Any, observed: Any) -> bool:
    expected = _reference_value(reference)
    if expected is None or observed is None:
        return expected is None and observed is None
    if isinstance(expected, float) or isinstance(observed, float):
        return math.isclose(
            float(observed),
            float(expected),
            rel_tol=EXACT_REL_TOLERANCE,
            abs_tol=EXACT_ABS_TOLERANCE,
        )
    return observed == expected


def _openeye_mol(smiles: str) -> Any:
    from openeye import oechem

    mol = oechem.OEGraphMol()
    if not oechem.OESmilesToMol(mol, smiles):
        raise ValueError(f"Could not parse SMILES: {smiles}")
    return mol


def _format_difference(
    descriptor: str,
    smiles: str,
    reference: Any,
    observed: Any,
    definition: Mapping[str, Any],
) -> str:
    primitive = definition.get("source_type", "")
    return (
        f"{descriptor} {smiles}: reference={reference!r} observed={observed!r} "
        f"primitive={primitive}"
    )


def _format_deferred_concrete_value(
    descriptor: str,
    smiles: str,
    observed: Any,
    definition: Mapping[str, Any],
) -> str:
    primitive = definition.get("source_type", "")
    return (
        f"{descriptor} {smiles}: deferred descriptor produced concrete value "
        f"{observed!r} primitive={primitive}"
    )


def _compare(
    references: Mapping[str, Any],
    policy: Mapping[str, Any],
    selected: Sequence[str],
) -> tuple[dict[str, int], list[str]]:
    import oefp

    descriptor_names = _definition_names(references)
    definitions = _definitions_by_name(references)
    descriptor_policies = _descriptor_policy(policy)
    row_policies = _row_divergence_policy(policy)
    counts = {
        "exact": 0,
        "accepted_divergences": 0,
        "deferred": 0,
        "not_applicable": 0,
        "unclassified": 0,
    }
    unclassified: list[str] = []

    for row in references["reference_rows"]:
        smiles = str(row["smiles"])
        expected_by_name = _reference_values_by_name(descriptor_names, row)
        descriptors = oefp.mordred_descriptors(_openeye_mol(smiles))

        for name in selected:
            expected = expected_by_name[name]
            observed = descriptors[name]
            descriptor_policy = descriptor_policies.get(name, {})
            status = descriptor_policy.get("status")
            if status == "deferred":
                if observed is None:
                    counts["deferred"] += 1
                else:
                    counts["unclassified"] += 1
                    unclassified.append(
                        _format_deferred_concrete_value(
                            name,
                            smiles,
                            observed,
                            definitions[name],
                        )
                    )
            elif status == "not_applicable":
                if _is_absent_reference_value(expected) and observed is None:
                    counts["not_applicable"] += 1
                else:
                    counts["unclassified"] += 1
                    unclassified.append(
                        _format_difference(
                            name,
                            smiles,
                            expected,
                            observed,
                            definitions[name],
                        )
                    )
            elif _values_match(expected, observed):
                counts["exact"] += 1
            elif (name, smiles) in row_policies:
                counts["accepted_divergences"] += 1
            else:
                counts["unclassified"] += 1
                unclassified.append(
                    _format_difference(
                        name,
                        smiles,
                        expected,
                        observed,
                        definitions[name],
                    )
                )

    return counts, unclassified


def _print_counts(counts: Mapping[str, int]) -> None:
    print(f"exact matches: {counts['exact']}")
    print(f"accepted divergences: {counts['accepted_divergences']}")
    print(f"deferred descriptors: {counts['deferred']}")
    print(f"not-applicable descriptors: {counts['not_applicable']}")
    print(f"unclassified differences: {counts['unclassified']}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare OEFP Mordred descriptor output with static references."
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
        help="Path to mordred_divergences.json.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=(),
        help="Mordred source_type families to compare.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=(),
        help="Descriptor names to compare.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compare every descriptor in the static reference fixture.",
    )
    parser.add_argument(
        "--max-differences",
        type=int,
        default=50,
        help="Maximum number of unclassified difference examples to print.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    references = _load_json(REFERENCE_FIXTURE)
    policy = _load_json(args.policy)
    selected = _selected_names(references, args.families, args.names, args.all)

    try:
        counts, unclassified = _compare(references, policy, selected)
    except ImportError as exc:
        print(f"Required runtime dependency is unavailable: {exc}", file=sys.stderr)
        return 2

    print(f"selected descriptors: {len(selected)}")
    print(f"reference rows: {len(references['reference_rows'])}")
    _print_counts(counts)
    if unclassified:
        print("\nunclassified differences:", file=sys.stderr)
        for difference in unclassified[: args.max_differences]:
            print(f"- {difference}", file=sys.stderr)
        remaining = len(unclassified) - args.max_differences
        if remaining > 0:
            print(f"... {remaining} more unclassified differences omitted", file=sys.stderr)
    return 1 if counts["unclassified"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
