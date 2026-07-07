"""Generate the RDKit 2D descriptor schema and reference fixtures.

RDKit is a test-time conformance oracle only. This script introspects
``rdkit.Chem.Descriptors._descList`` to emit the static C++ schema array and
per-molecule reference values over a fixed SMILES panel.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

EXPECTED_RDKIT_VERSION = "2026.03.3"

# Reused verbatim from generate_mordred_references.py for cross-source panel
# comparability, then extended with fragment/VSA coverage molecules.
SMILES_PANEL = [
    "CCO", "c1ccncc1", "CC(C)(C)Cl", "O=S(=O)(N)C1=CC=CC=C1", "CI", "C[Na]",
    "C[N+](C)(C)CC(=O)[O-]", "C1=CC2=C(C=C1)C=CC=C2", "C12C3C4C1C5C2C3C45",
    "C1C2CC3CC1CC(C2)C3", "CC#N", "FC(F)(F)c1ccc(Br)cc1", "CN",
    "c1ccccc1[N+](=O)[O-]", "CC(=O)O", "OP(=O)(O)O", "C", "CC", "CCC",
    "CCCCCC", "C1CCCCC1", "C1CC1C", "CCCCCCCCCCCCCCCC", "O=[Se]=O", "[13CH4]",
    "COC(=O)c1ccc(OCC)c(O)c1C(=O)OCC",
]

# Molecules chosen so every fr_* SMARTS and every VSA bin is exercised
# non-trivially. Curated during Task 9 (fragments) and Task 8 (VSA); seeded here
# with a broad drug-like set and expanded as coverage assertions demand.
RDKIT_COVERAGE_SUPPLEMENT = [
    "O=C(Nc1ccc(cc1)S(=O)(=O)N)c1ccccc1Cl",   # amide, sulfonamide, aryl halide
    "CC(=O)Oc1ccccc1C(=O)O",                    # ester + carboxylic acid (aspirin)
    "c1ccc2[nH]ccc2c1",                         # indole (aromatic N-H)
    "OCC(O)CO",                                 # polyols
    "CCN(CC)CCOC(=O)c1ccccc1N",                 # amine, ester, aniline
    "O=C1CCCCC1",                               # ketone
    "N#Cc1ccccc1",                              # nitrile
    "CS(=O)(=O)c1ccccc1",                       # sulfone
    "OB(O)c1ccccc1",                            # boronic acid
    "[O-][N+](=O)c1ccc(cc1)N=Nc1ccccc1",        # nitro + azo
    # VSA bin coverage (SMR, EState, SlogP bins)
    "c1ccc2c(c1)c1c(c3c2cccc3)cccc1",           # large PAH for VSA10
    "c1ccc(cc1)c1ccc(cc1)c1ccc(cc1)c1ccccc1",   # quaterphenyl for VSA9
    "C1CCC2C1CCC1C2CCC2C1CCCC2",                # steroid skeleton
    "C=N",                                      # imine for SMR_VSA2
    "CN",                                       # methylamine for SMR_VSA4 (already have but ensure)
    # Fragment coverage (fr_*)
    "CSC",                                      # fr_C_S sulfide
    "CC=O",                                     # fr_aldehyde
    "OCCN1CCCCC1",                              # fr_HOCCN
    "C=NC",                                     # fr_Imine
    "CNO",                                      # fr_N_O
    "CN1CCCCC1",                                # fr_Ndealkylation2
    "CS",                                       # fr_SH thiol
    "CC(=O)NC(=O)C",                            # fr_imide
    "CCNC(=O)OC",                               # fr_alkyl_carbamate
    "CC=CCO",                                   # fr_allylic_oxid
    "CC(=N)N",                                  # fr_amidine (not guanidine)
    "Cc1ccccc1",                                # fr_aryl_methyl
    "CCN=[N+]=[N-]",                            # fr_azide (alkyl azide)
    "O=C1NC(=O)NC(=O)C1",                       # fr_barbitur
    "O=C1CN=C(c2ccccc2)c2ccccc2N1",             # fr_benzodiazepine (7-ring core)
    "[CH2-][N+]#N",                             # fr_diazo
    "C1=CNC=CC1",                               # fr_dihydropyridine
    "C1OC1",                                    # fr_epoxide
    "c1ccoc1",                                  # fr_furan
    "N=C(N)N",                                  # fr_guanido (guanidine)
    "NN",                                       # fr_hdrzine
    "C=NN",                                     # fr_hdrzone
    "c1cnc[nH]1",                               # fr_imidazole
    "N=C=O",                                    # fr_isocyan
    "N=C=S",                                    # fr_isothiocyan
    "O=C1CCN1",                                 # fr_lactam (beta-lactam)
    "C1COC(=O)C1",                              # fr_lactone
    "C1CNCCO1",                                 # fr_morpholine
    "CN=O",                                     # fr_nitroso
    "c1cnco1",                                  # fr_oxazole
    "C=NO",                                     # fr_oxime
    "CCOP(=O)(O)O",                             # fr_phos_ester
    "C1CCNCC1",                                 # fr_piperdine
    "C1CNCCN1",                                 # fr_piperzine
    "CC(N)=O",                                  # fr_priamide
    "C#C",                                      # fr_term_acetylene
    "c1nnn[nH]1",                               # fr_tetrazole
    "c1cncs1",                                  # fr_thiazole
    "SC#N",                                     # fr_thiocyan
    "c1ccsc1",                                  # fr_thiophene
    "NC(=O)N",                                  # fr_urea
    # CountsWeights integer-count regression coverage (bridgehead/spiro):
    "C1CCC2(CC1)CCC2",                          # spiro[4.5] center: BH=0, Spiro=1
    "[O]",                                       # atomic oxygen radical: NumRadicalElectrons=2
    # CountsWeights integer-count regression coverage (radical/valence):
    "C1CC23CCC(C1)(CC2)CC3",                     # bridged bicyclo: BH=2 (propellane-safe)
    "[Cu]",                                      # metal radical: NumRadicalElectrons=1 (parity branch)
    "[Fe+2]",                                    # metal skip: NumRadicalElectrons=0 (undefined-valence gate)
    # CountsWeights regression coverage (stereo bracket-H suppression): OpenEye
    # keeps the [C@H] hydrogen explicit while RDKit treats it as implicit; these
    # lock the local H-suppression so weights/valence/Morgan-density match RDKit.
    "C[C@H](N)C(=O)O",                           # L-alanine: bracket-H + real stereocenter
    "C[C@@H](O)C(=O)O",                          # lactic acid: bracket-H + stereocenter
    # Connectivity regression coverage (Chi valence-delta unsigned wrap): these
    # hypervalent hydride anions have a NEGATIVE Kier-Hall numerator (Zv - h),
    # which RDKit forms in unsigned 32-bit so it wraps to 2^32-1 and yields a
    # finite tiny Chi0n/Chi0v (~1.5e-05 / ~4.6e-05) rather than a NaN. They lock
    # the native unsigned-wrap reproduction against a regression to sqrt(<0).
    "[BH4-]",                                    # boron: Zv=3, h=4 -> wraps (Chi0n=Chi0v)
    "[AlH4-]",                                   # aluminium: 2nd-row denominator branch
    "[SiH5-]",                                   # silicon: 2nd-row denominator branch
]

# RDKit descriptors excluded from the schema because they are structurally
# always zero (empty VSA bins; RDKit's fixed bin boundaries never populate these
# ranges). Excluded per user decision 2026-07-06.
RDKIT_EXCLUDED_DESCRIPTORS = {"SMR_VSA8", "SlogP_VSA9", "EState_VSA11"}

# Curated cross-source identities. A name maps to a shared canonical_id ONLY
# after Task 4 verifies byte-identical output vs the existing source over the
# panel. Absent => empty => never deduplicated. Start conservative.
RDKIT_CANONICAL_IDS: dict[str, str] = {
    "HeavyAtomCount": "quantity:heavy_atom_count",
    # ExactMolWt intentionally NOT tagged: the RDKit source computes exact weight
    # on an H-suppressed molecule (matching RDKit's implicit-H model) while the
    # OpenEye and Mordred sources compute weight on the raw molecule. On bracket-H
    # molecules like C[C@H](N)C(=O)O (L-alanine), RDKit reports 89.0477 while the
    # others report 90.0555. This is not a bug; it reflects the different input
    # representations (H-suppressed vs unsuppressed) post-bracket-H unification.
    # MolWt, TPSA, NumHDonors, NumHAcceptors intentionally NOT tagged until a
    # byte-identical shared-helper path is proven.
}

# name -> "exact" | "tight" | "loose", each with a one-line rationale comment.
# Assigned by fidelity class (spec §2.5); refined against measured deviation but
# only loosened with a recorded rationale. Seeded per family as tasks land.
RDKIT_TOLERANCE_TIERS: dict[str, str] = {
    # Counts/weights (Task 3): exact — same atomic constants / graph counts.
    "HeavyAtomCount": "exact",
    # ExactMolWt re-curated exact->tight (Task 3): OpenEye and RDKit ship slightly
    # different isotope-mass constants, so the native monoisotopic weight differs
    # from the oracle by ~3e-8 relative on iodine/sodium molecules (CI, C[Na]).
    # This is a constant-table difference, not a computation bug; tight (1e-4)
    # covers it with wide margin while still catching real errors.
    "ExactMolWt": "tight",
    "NumValenceElectrons": "exact",
    "NumHeteroatoms": "exact",
    "FractionCSP3": "exact",
    "MolWt": "tight",            # average MW: element-average table rounding
    "HeavyAtomMolWt": "tight",
    # Ring counts (Task 5): all integer counts, so the conformance test compares
    # them for exact equality and the tier is only advisory. Marked exact because
    # every one matches RDKit exactly across the panel: RingCount reproduces
    # RDKit's symmetrized SSSR count (via the shared ring-perception engine, not a
    # cyclomatic number), and OpenEye's aromaticity model agrees with RDKit on the
    # aromatic/aliphatic/saturated splits for every panel molecule.
    "RingCount": "exact",
    "NumAromaticRings": "exact",
    "NumAliphaticRings": "exact",
    "NumSaturatedRings": "exact",
    "NumAromaticCarbocycles": "exact",
    "NumAromaticHeterocycles": "exact",
    "NumAliphaticCarbocycles": "exact",
    "NumAliphaticHeterocycles": "exact",
    "NumSaturatedCarbocycles": "exact",
    "NumSaturatedHeterocycles": "exact",
    "NumHeterocycles": "exact",
    # Connectivity (Task 6): float connectivity/shape indices. The native port
    # reproduces RDKit's definitions from the shared heavy-atom graph and matches
    # the oracle to ~1e-15 across the panel (well inside tight/1e-4), so the Chi
    # family, HallKierAlpha, the Kappa shape indices, BertzCT and BalabanJ are
    # tight. Ipc/AvgIpc are tight because their characteristic-polynomial +
    # log-summation ordering is sensitive to accumulation order; the native
    # Le Verrier-Faddeev evaluation still tracks the oracle to ~1e-16 here.
    "Chi0": "tight",
    "Chi1": "tight",
    "Chi0n": "tight",
    "Chi1n": "tight",
    "Chi2n": "tight",
    "Chi3n": "tight",
    "Chi4n": "tight",
    "Chi0v": "tight",
    "Chi1v": "tight",
    "Chi2v": "tight",
    "Chi3v": "tight",
    "Chi4v": "tight",
    "HallKierAlpha": "tight",
    "Kappa1": "tight",
    "Kappa2": "tight",
    "Kappa3": "tight",
    "BertzCT": "tight",
    "BalabanJ": "tight",
    "Ipc": "tight",
    "AvgIpc": "tight",
    # Phi (Task 6): CountsWeights column = Kappa1*Kappa2/heavy-count.
    "Phi": "tight",
    # ... remaining families appended by their tasks.
}

TIER_DEFAULT = "loose"  # any descriptor not yet classified is loosely gated
                        # AND excluded from the conformance ENABLED set until a
                        # task assigns it a tier and enables it.


def _apply_proxy() -> None:
    for key, value in {
        "HTTP_PROXY": "http://proxy-server.bms.com:8080",
        "HTTPS_PROXY": "http://proxy-server.bms.com:8080",
    }.items():
        os.environ.setdefault(key, value)


def _rdkit_version() -> str:
    import rdkit
    return str(rdkit.__version__)


def _value_kind(name: str) -> str:
    # Integer-valued RDKit descriptors: counts and fragment matches.
    if name.startswith("fr_") or name.startswith("Num") or name.startswith("NO") \
            or name.startswith("NHOH") or name in {"HeavyAtomCount", "RingCount"}:
        return "int"
    return "float"


def _group(name: str) -> str:
    if name.startswith("fr_"):
        return "rdkit:Fragments"
    if "VSA" in name:
        return "rdkit:VSA"
    if name.startswith("BCUT2D"):
        return "rdkit:BCUT2D"
    if name.endswith("EStateIndex"):
        return "rdkit:EState"
    if name.endswith("PartialCharge"):
        return "rdkit:PartialCharge"
    if name in {"MolLogP", "MolMR"}:
        return "rdkit:Crippen"
    if name in {"TPSA", "LabuteASA"}:
        return "rdkit:SurfacePolarity"
    if name == "qed":
        return "rdkit:Composite"
    if name.startswith(("Chi", "Kappa", "HallKierAlpha")) or name in {
            "BertzCT", "BalabanJ", "Ipc", "AvgIpc"}:
        return "rdkit:Connectivity"
    if "Ring" in name or name.startswith("Num") and "cycle" in name.lower():
        return "rdkit:RingCounts"
    return "rdkit:CountsWeights"


def _json_value(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    number = float(value)
    if not math.isfinite(number):
        return {"error_type": str(number), "state": "nonfinite"}
    return number


def _reference_payload(descriptor_source_id: str, allow_version_mismatch: bool = False) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    desc_list = list(Descriptors._descList)  # [(name, fn), ...]
    version = _rdkit_version()

    # Guard against unexpected RDKit version to preserve deterministic output
    if version != EXPECTED_RDKIT_VERSION:
        msg = f"Expected RDKit {EXPECTED_RDKIT_VERSION}, imported {version!r}."
        if allow_version_mismatch:
            print(f"WARNING: {msg}")
        else:
            raise RuntimeError(msg)

    source_version = f"RDKit-{version}"
    panel = SMILES_PANEL + RDKIT_COVERAGE_SUPPLEMENT

    # Filter out excluded descriptors from both definitions and per-row values
    desc_list = [(name, fn) for name, fn in desc_list if name not in RDKIT_EXCLUDED_DESCRIPTORS]

    definitions = []
    for name, _fn in desc_list:
        definitions.append({
            "name": name,
            "group": _group(name),
            "value_kind": _value_kind(name),
            "source_name": "RDKit",
            "source_type": _group(name).split(":", 1)[1],
            "source_version": source_version,
            "parameters": "",
            "units": "",
            "description": name,
            "tier": RDKIT_TOLERANCE_TIERS.get(name, TIER_DEFAULT),
            "canonical_id": RDKIT_CANONICAL_IDS.get(name, ""),
            "prerequisites": 0,  # all 2D; no coordinate prerequisite
        })

    rows = []
    for smiles in panel:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
        values = []
        for name, fn in desc_list:
            try:
                values.append(_json_value(fn(mol)))
            except Exception as exc:  # RDKit raises on a few degenerate inputs
                values.append({"error_type": type(exc).__name__, "state": "error"})
        rows.append({"smiles": smiles, "values": values})

    _assert_coverage(definitions, rows)

    return {
        "schema_id": _schema_id(definitions),
        "source": {"descriptor_source": descriptor_source_id, "name": "RDKit",
                   "version": source_version},
        "smiles_panel": panel,
        "excluded_descriptors": list(sorted(RDKIT_EXCLUDED_DESCRIPTORS)),
        "definitions": definitions,
        "reference_rows": rows,
    }


def _assert_coverage(definitions: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    names = [d["name"] for d in definitions]
    idx = {name: i for i, name in enumerate(names)}

    def nonzero_somewhere(name: str) -> bool:
        i = idx[name]
        for row in rows:
            v = row["values"][i]
            if isinstance(v, (int, float)) and v != 0:
                return True
        return False

    # Hard assertion: every VSA bin in the schema must fire at least once.
    # (Always-zero bins are excluded from the schema at enumeration time.)
    for name in names:
        if "VSA" in name:
            if not nonzero_somewhere(name):
                raise AssertionError(
                    f"VSA coverage gap: {name} is zero; add a molecule that "
                    f"activates this bin to RDKIT_COVERAGE_SUPPLEMENT.")

    # Hard assertion: every fr_* descriptor must fire at least once.
    # All RDKit fr_* patterns are chemically calculable; expand the panel to
    # cover any that don't yet have an activator molecule.
    for name in names:
        if name.startswith("fr_"):
            if not nonzero_somewhere(name):
                raise AssertionError(
                    f"Fragment coverage gap: {name} is zero; add a molecule to "
                    f"RDKIT_COVERAGE_SUPPLEMENT that activates it.")

    # Soft warnings: any descriptor all-zero/all-constant across the panel.
    for name in names:
        i = idx[name]
        seen = {tuple(sorted(v.items())) if isinstance(v, dict) else v
                for v in (row["values"][i] for row in rows)}
        if len(seen) == 1:
            print(f"WARNING: {name} is constant across the panel (weak signal).")


def _schema_id(definitions: list[dict[str, Any]]) -> str:
    # FNV-1a over the same field layout generate_mordred_references.py uses, so
    # the C++ DescriptorSchema::SchemaId() matches. Copy that _schema_id body
    # verbatim (it excludes `tier`, which is not a schema field).
    def append_field(text: str, value: str) -> str:
        return f"{text}{len(value.encode('utf-8'))}:{value}"
    serialized = "oefp-descriptor-schema-v1\n"
    for d in definitions:
        serialized = append_field(serialized, d["name"])
        serialized += "|"
        serialized += d["value_kind"]
        serialized += "|"
        serialized = append_field(serialized, d["group"])
        serialized += "|"
        serialized = append_field(serialized, d["source_name"])
        serialized += "|"
        serialized = append_field(serialized, d["source_type"])
        serialized += "|"
        serialized = append_field(serialized, d["source_version"])
        serialized += "|"
        serialized = append_field(serialized, d["parameters"])
        serialized += "|"
        serialized = append_field(serialized, d.get("units", ""))
        serialized += "|"
        serialized = append_field(serialized, d["description"])
        serialized += "|"
        serialized = append_field(serialized, d.get("canonical_id", ""))
        serialized += "|"
        serialized += str(int(d.get("prerequisites", 0)))
        serialized += "|-\n"
    value = 14695981039346656037
    for byte in serialized.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def _cpp_kind(value_kind: str) -> str:
    return {"bool": "DescriptorValueKind::Bool", "int": "DescriptorValueKind::Int",
            "float": "DescriptorValueKind::Float",
            "string": "DescriptorValueKind::String"}[value_kind]


def _write_cpp_schema(payload: dict[str, Any], output: Path) -> None:
    # Mirror generate_mordred_references._write_cpp_schema exactly, swapping
    # Mordred->RDKit names and the source_version literal. The tier field is NOT
    # emitted to C++.
    defs = payload["definitions"]
    src_version = payload["source"]["version"]
    lines = [
        '#include "oefp/rdkit_descriptors.h"', "", "#include <array>",
        "#include <memory>", "#include <string>", "#include <utility>",
        "#include <vector>", "", "namespace OEFP {", "namespace {", "",
        "struct StaticRDKitDefinition {", "    const char* name;",
        "    DescriptorValueKind value_kind;", "    const char* group;",
        "    const char* source_type;", "    const char* parameters;",
        "    const char* description;", "    const char* canonical_id;",
        "    DescriptorPrerequisites prerequisites;", "};", "",
        f"constexpr std::array<StaticRDKitDefinition, {len(defs)}> kRDKitDefinitions{{{{",
    ]
    for d in defs:
        lines.append(
            "    {" f"{json.dumps(d['name'])}, {_cpp_kind(d['value_kind'])}, "
            f"{json.dumps(d['group'])}, {json.dumps(d['source_type'])}, "
            f"{json.dumps(d['parameters'])}, {json.dumps(d['description'])}, "
            f"{json.dumps(d.get('canonical_id', ''))}, "
            f"{int(d.get('prerequisites', 0))}u" "},")
    lines += [
        "}};", "", "} // namespace", "",
        "std::shared_ptr<const DescriptorSchema> RDKitDescriptorSchema() {",
        "    static const std::shared_ptr<const DescriptorSchema> schema = [] {",
        "        std::vector<DescriptorDefinition> definitions;",
        "        definitions.reserve(kRDKitDefinitions.size());",
        "        for (const auto& item : kRDKitDefinitions) {",
        "            DescriptorDefinition definition;",
        "            definition.name = item.name;",
        "            definition.value_kind = item.value_kind;",
        "            definition.group = item.group;",
        '            definition.source_name = "RDKit";',
        "            definition.source_type = item.source_type;",
        f'            definition.source_version = "{src_version}";',
        "            definition.parameters = item.parameters;",
        "            definition.description = item.description;",
        "            definition.canonical_id = item.canonical_id;",
        "            definition.prerequisites = item.prerequisites;",
        "            definitions.push_back(std::move(definition));",
        "        }",
        "        return std::make_shared<const DescriptorSchema>(std::move(definitions));",
        "    }();",
        "    return schema;",
        "}", "", "} // namespace OEFP", "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _apply_proxy()
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor-source",
                        default=os.environ.get("OEFP_RDKIT_SOURCE_ID", "rdkit-2026.03.3"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpp-output", type=Path)
    parser.add_argument("--allow-rdkit-version-mismatch", action="store_true",
                        help="Downgrade version mismatch from error to warning")
    args = parser.parse_args()

    payload = _reference_payload(args.descriptor_source, args.allow_rdkit_version_mismatch)
    # 217 in RDKit _descList minus 3 always-zero VSA bins = 214 schema descriptors
    if len(payload["definitions"]) != 214:
        raise RuntimeError(f"Expected 214 RDKit definitions, got {len(payload['definitions'])}.")
    for row in payload["reference_rows"]:
        if len(row["values"]) != 214:
            raise RuntimeError(f"Expected 214 values for {row['smiles']}.")

    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    if args.cpp_output is not None:
        _write_cpp_schema(payload, args.cpp_output)


if __name__ == "__main__":
    main()
