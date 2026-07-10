# OEFP

High-performance molecular fingerprints for the [OpenEye Toolkits](https://www.eyesopen.com/).

OEFP generates RDKit-compatible Morgan and topological Atom Pair fingerprints
from OpenEye molecules, stores them in compact C++ containers, and compares
them with fast scalar and batch kernels. It also provides Morgan and
topological Atom Pair descriptor rows plus Mordred-compatible and
RDKit-compatible descriptors.

OEFP currently supports dense binary, sparse binary, and sparse counted
fingerprint containers; scalar comparison; query-to-batch comparison; `cdist`;
SciPy-compatible condensed `pdist`; and
Arrow/Parquet interchange for schema-backed descriptor rows.

Try it out:

```bash
pip install oefp
```

## Usage

Here are a few examples of using `oefp`.

### Python

```python
from openeye import oechem
import oefp

mol = oechem.OEGraphMol()
oechem.OESmilesToMol(mol, "CC(=O)OC1=CC=CC=C1C(=O)O")  # aspirin

# Generate an RDKit-compatible Morgan fingerprint.
fp = oefp.morgan_fingerprint(mol, radius=2, num_bits=2048)
print(fp.popcount)
print(fp.words[:4])

# Compare fingerprints.
score = oefp.compare(fp, fp, oefp.Metric.tanimoto())
print(score)
```

Use reusable generators when applying the same options to many molecules:

```python
from openeye import oechem
import oefp

smiles = ["c1ccccc1", "c1ccc(O)cc1", "CC(=O)O"]
mols = []
for smi in smiles:
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, smi)
    mols.append(mol)

generator = oefp.MorganGenerator(radius=2, num_bits=2048)
fps = [generator.fingerprint(mol) for mol in mols]

batch = oefp.OEFPBatch.from_fingerprints(fps)
distances = oefp.pdist(batch, oefp.Metric.jaccard())
```

Build a batch directly from molecules in one call:

```python
batch = oefp.OEFPBatch.from_molecules(mols, oefp.morgan_fingerprint, radius=2)
distances = oefp.pdist(batch, oefp.Metric.jaccard())
```

`from_molecules` is available on `OEFPBatch`, `OEFPCountBatch`,
`OEFPSparseBatch`, and `DescriptorBatch`; pass the matching generator
(`morgan_count_fingerprint`, `morgan_sparse_fingerprint`, `morgan_descriptors`,
the Atom Pair / Topological Torsions functions, …) and any keyword options.

Generate sparse and counted fingerprints:

```python
folded_count = oefp.morgan_count_fingerprint(mol)
sparse_binary = oefp.morgan_sparse_fingerprint(mol)
atom_pair_count = oefp.atom_pair_sparse_count_fingerprint(mol)

print(folded_count.indices[:5])
print(folded_count.counts[:5])
print(sparse_binary.indices[:5])
print(atom_pair_count.total_count)
```

Inspect Morgan bit provenance:

```python
result = oefp.morgan_fingerprint_with_mapping(mol)
print(result.fingerprint.popcount)
print(result.mapping.bit_info())
```

Import and export OpenEye fingerprints:

```python
from openeye import oechem, oegraphsim
import oefp

mol = oechem.OEGraphMol()
oechem.OESmilesToMol(mol, "CCO")

oe_fp = oegraphsim.OEFingerPrint()
oegraphsim.OEMakeCircularFP(oe_fp, mol)

fp = oefp.from_openeye_fingerprint(oe_fp)
round_tripped = oefp.to_openeye_fingerprint(fp)
print(oegraphsim.OETanimoto(oe_fp, round_tripped))
```

Work with Mordred-compatible named descriptors:

```python
from openeye import oechem
import oefp

mol = oechem.OEGraphMol()
oechem.OESmilesToMol(mol, "CCO")

row = oefp.mordred_descriptors(mol)
schema = row.schema

print(schema.schema_id)
print(row["MW"])
print(row["GeomDiameter"])  # None unless the input already has 3D coordinates.

requires_3d = [
    definition.name
    for definition in schema.definitions
    if definition.prerequisites & oefp.DESCRIPTOR_PREREQUISITE_COORDINATES_3D
]
print(len(requires_3d))
```

Descriptor calculation never generates 2D or 3D coordinates implicitly. When a
descriptor requires coordinates that the input molecule does not already have,
that descriptor value remains missing (`None` in Python).

Work with RDKit-compatible named descriptors:

```python
from openeye import oechem
import oefp

mol = oechem.OEGraphMol()
oechem.OESmilesToMol(mol, "CC(=O)OC1=CC=CC=C1C(=O)O")  # aspirin

row = oefp.rdkit_descriptors(mol)
schema = row.schema

print(len(schema.names))        # 213 native RDKit 2D descriptors
print(row["MolWt"])             # 180.157...
print(row["TPSA"])              # 63.6
print(row["NumAromaticRings"])  # 1
print(row["MolLogP"])           # Wildman-Crippen SLogP
print(row["qed"])               # 0.550...
```

OEFP reproduces 213 of RDKit's 2D descriptors natively, matched to RDKit
2026.03.3 within per-descriptor tolerance tiers. RDKit is used only as a
test-time conformance oracle; it is not a runtime dependency. Four of RDKit's
217 descriptors are excluded from the schema: three structurally-always-zero
VSA bins (`SMR_VSA8`, `SlogP_VSA9`, `EState_VSA11`) and `SPS` (`SpacialScore`),
whose exact value depends on RDKit's aromaticity perception, which differs from
OpenEye's on some conjugated ring systems.

Use `RDKitDescriptorSource` in a calculator to compute batches or select a
subset of columns:

```python
calc = oefp.DescriptorCalculator([oefp.RDKitDescriptorSource()])

smiles = ["c1ccccc1", "c1ccc(O)cc1", "CC(=O)O"]
mols = []
for smi in smiles:
    m = oechem.OEGraphMol()
    oechem.OESmilesToMol(m, smi)
    mols.append(m)

batch = calc.calculate_batch(mols)
print(batch.size)               # 3
print(batch[0]["MolLogP"], batch[0]["NumHAcceptors"])
```

The Mordred and RDKit sources share many descriptor names (for example
`BalabanJ`, `Chi0`, and `TPSA`). Registering both in one calculator without
narrowing raises a name-collision error, since the same name would appear
twice. Select the columns you want from one source to combine them:

```python
calc = oefp.DescriptorCalculator([
    oefp.OpenEyePropertyDescriptorSource(),
    (oefp.RDKitDescriptorSource(), ["MolLogP", "TPSA", "qed"]),
])
print("MolLogP" in calc.schema.names)  # True
```

Compute merged, deduplicated descriptors from multiple sources:

```python
from openeye import oechem
import oefp

# Build a descriptor calculator from Mordred and OpenEye property sources.
calc = oefp.DescriptorCalculator([
    oefp.MordredDescriptorSource(),
    oefp.OpenEyePropertyDescriptorSource(),
])

# The schema deduplicates by canonical_id with first-wins by registration
# order. Mordred is registered first, so its MW and nHBAcc are kept and
# OpenEye's MolecularWeight and HBA are dropped (same canonical_id). XLogP
# survives because it has no Mordred equivalent.
print(len(calc.schema.names))
print("MW" in calc.schema.names)         # True (Mordred kept)
print("MolecularWeight" in calc.schema.names)  # False (OpenEye duplicate dropped)
print("HBA" in calc.schema.names)        # False (dedup: Mordred nHBAcc wins)
print("XLogP" in calc.schema.names)      # True (OpenEye-unique survives)

# Build molecules.
smiles = ["c1ccccc1", "c1ccc(O)cc1", "CC(=O)O"]
mols = []
for smi in smiles:
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, smi)
    mols.append(mol)

# Compute a batch.
batch = calc.calculate_batch(mols)
print(batch.size)
print(list(batch.schema.names)[:5])
```

Work with `DescriptorBatch` sugar:

```python
# Iterate rows as {name: value} dictionaries.
for row in batch:
    print(row["MW"], row["XLogP"])

# Or get the first row.
print(batch[0]["MW"])

# Convert to column-oriented form.
columns = batch.to_dict()
print(columns["MW"])  # [value1, value2, value3, ...]

# Convert to row-oriented list of dicts.
rows = batch.to_records()
print(rows[0])

# Vertically concatenate two batches with the same schema.
combined = batch + batch
print(combined.size)
```

Work with kallisto atom and bond descriptors:

```python
from openeye import oechem
import oefp

# Load a molecule with pre-existing 3D coordinates.
mol = oechem.OEGraphMol()
ifs = oechem.oemolistream("tests/data/kallisto_panel/ethane.sdf")
oechem.OEReadMolecule(ifs, mol)

# Compute per-atom descriptors: coordination numbers (cn_erf, cn_cov, cn_exp),
# proximity (prox), EEQ partial charge (eeq), dynamic polarizability (alp),
# and van der Waals radii (vdw_rahm, vdw_truhlar).
atom_result = oefp.kallisto_atom_descriptors(mol)
print(atom_result.atom_count)  # 8 atoms in ethane
print(atom_result["eeq"])  # Array of EEQ partial charges in atomic units (e)
print(atom_result["cn_erf"])  # erf coordination numbers (dimensionless)
print(atom_result.eeq)  # Column access via attribute

# Compute per-bond descriptors: Sterimol L/B1/B5 (bond length and cross-sections).
bond_result = oefp.kallisto_bond_descriptors(mol)
print(bond_result.bond_count)  # 14 directed bonds in ethane
print(bond_result["sterimol_L"])  # Sterimol L values in Bohr

# Access schema information.
atom_schema = oefp.kallisto_atom_schema()
bond_schema = oefp.kallisto_bond_schema()
print(atom_schema.names)  # ('cn_erf', 'cn_cov', 'cn_exp', 'prox', 'eeq', 'alp', 'vdw_rahm', 'vdw_truhlar')
print(bond_schema.names)  # ('sterimol_L', 'sterimol_B1', 'sterimol_B5')

# Compute Sterimol for a specific bond (origin and partner are atom indices).
origin = 0  # First carbon in ethane
partner = 1  # Second carbon in ethane
L, B1, B5 = oefp.sterimol(mol, origin, partner)
print(f"Sterimol L={L:.3f} B1={B1:.3f} B5={B5:.3f}")  # Values in Bohr

# Batch compute for multiple molecules.
mols = []
for sdf_name in ["methane.sdf", "ethane.sdf", "methanethiol.sdf"]:
    m = oechem.OEGraphMol()
    ifs = oechem.oemolistream(f"tests/data/kallisto_panel/{sdf_name}")
    oechem.OEReadMolecule(ifs, m)
    mols.append(m)

atom_batch = oefp.kallisto_atom_descriptors_batch(mols)
bond_batch = oefp.kallisto_bond_descriptors_batch(mols)
print(len(atom_batch))  # 3 segments (one per molecule)
print(len(atom_batch[0]["eeq"]))  # 5 atoms in methane
print(len(bond_batch[1]["sterimol_L"]))  # 14 directed bonds in ethane
```

Kallisto atom and bond descriptors require molecules with pre-existing 3D
coordinates. OEFP never generates coordinates; if a molecule lacks 3D
coordinates or contains any atom with atomic number greater than 86, that
molecule is skipped and yields an empty result for both atom and bond
descriptors.

All descriptor values are returned in kallisto's native atomic units:
- Coordination numbers (cn_erf, cn_cov, cn_exp) and proximity (prox) are dimensionless
- EEQ partial charges (eeq) are in elementary charge units (e)
- Dynamic polarizabilities (alp) are in cubic Bohr (Bohr^3)
- van der Waals radii (vdw_rahm, vdw_truhlar) and Sterimol parameters (L, B1, B5) are in Bohr

The kallisto port reproduces kallisto 1.0.10 parameter tables and numeric
methods. kallisto is used as a test-time conformance oracle only and is not a
runtime dependency of OEFP. For more information about kallisto, see
https://github.com/AstraZeneca/kallisto

### C++

```cpp
#include <oefp/oefp.h>
#include <oechem.h>
#include <iostream>

int main() {
    OEChem::OEGraphMol mol_a;
    OEChem::OEGraphMol mol_b;
    OEChem::OESmilesToMol(mol_a, "c1ccccc1");
    OEChem::OESmilesToMol(mol_b, "c1ccc(O)cc1");

    OEFP::MorganGenerator generator;
    OEFP::OEFP fp_a = generator.Fingerprint(mol_a);
    OEFP::OEFP fp_b = generator.Fingerprint(mol_b);

    double score = OEFP::Compare(fp_a, fp_b, OEFP::Metric::Tanimoto());
    std::cout << score << "\n";

    return 0;
}
```

## Supported Fingerprints

| Family | Outputs | Notes |
|--------|---------|-------|
| Morgan | Folded binary, folded count, sparse binary, sparse count | Bit mapping is available for all Morgan outputs |
| Topological Atom Pair | Folded binary, folded count, sparse binary, sparse count | Uses connectivity distances; legacy `atom_pair_*` names remain compatibility aliases |
| OpenEye | `OEFingerPrint` import/export | Numeric type metadata is preserved when available |

## Supported Descriptors

| Family | Output | Notes |
|--------|--------|-------|
| Morgan | Raw counted integer-key descriptors | Uses unfurled Morgan environment identifiers |
| Topological Atom Pair | Raw counted string-key descriptors | Uses graph shortest-path distances and requires no coordinate generation |
| Distance Atom Pair | Reserved | Requires existing 3D coordinates and is not implemented yet |
| Mordred-compatible | Schema-backed named descriptor rows | Full Mordred 1.2.0 schema with implemented values filled and unsupported or unavailable values left missing |
| RDKit-compatible | Schema-backed named descriptor rows | 213 of RDKit's 2D descriptors reproduced natively and matched to RDKit within tolerance |
| Kallisto atom and bond | Per-atom and per-bond geometric descriptors | Coordination numbers, EEQ partial charges, dynamic polarizabilities, van der Waals radii, and Sterimol parameters; requires pre-existing 3D coordinates |

Morgan supports both ECFP-style and FCFP-style pharfingerprints
via `use_features=True` (Donor, Acceptor, Aromatic, Halogen, Basic, Acidic), 
matching RDKit's feature.
`morgan_fingerprint(mol, radius=2)` is ECFP4; `morgan_fingerprint(mol,
radius=2, use_features=True)` is FCFP4.

Morgan, Topological Atom Pair, and Topological Torsions outputs support
RDKit-compatible chirality encoding with `use_chirality=True` directly
from OpenEye molecule objects. There may be some differences in chirality
encoding between OpenEye and RDKit based on how each toolkit represents
stereo information.

## Installation

Install OpenEye Toolkits first:

```bash
pip install --extra-index-url https://pypi.anaconda.org/openeye/simple openeye-toolkits
```

Install OEFP:

```bash
pip install oefp
```

## Build from Source

Set the OpenEye C++ SDK path:

```bash
export OPENEYE_ROOT=/path/to/openeye/sdk
```

Build the C++ library and Python bindings:

```bash
cmake --preset debug
cmake --build build-debug
```

Install the Python package in editable mode:

```bash
pip install --config-settings editable_mode=compat -e python/
```

The `editable_mode=compat` flag keeps the package on a traditional editable
path that works with compiled SWIG extension modules.

## Tests

C++ tests:

```bash
cmake --build build-debug --target oefp_tests
ctest --test-dir build-debug --output-on-failure
```

Python tests:

```bash
PYTHONPATH=python python -m pytest tests/python -q
```

RDKit is required for conformance tests but is not a runtime dependency.

## Documentation

Build the Sphinx documentation:

```bash
python -m pip install -r docs/requirements.txt
make -C docs html
```

Open the local build:

```bash
open docs/_build/html/index.html
```

The documentation includes installation, quickstart, Python API notes, C++ API
reference generation through Doxygen, and release build guidance.

## Benchmarks

Run the RDKit generation and dense `pdist` benchmark:

```bash
PYTHONPATH=python python benchmarks/benchmark_rdkit_generation.py \
  --max-mols 1500 \
  --trials 7 \
  --warmup 1 \
  --pdist-size 400 \
  --generation-max-ratio 1.10 \
  --atom-pair-generation-max-ratio 1.10
```

Run the optional C++ guardrail against a local `oecluster` checkout:

```bash
cmake -S . -B build-bench \
  -DOEFP_BUILD_BENCHMARKS=ON \
  -DOEFP_OECLUSTER_SOURCE_DIR=/path/to/oecluster
cmake --build build-bench --target oefp_oecluster_fingerprint_benchmark
./build-bench/benchmarks/oefp_oecluster_fingerprint_benchmark 512 0 256
```

## Tools

| Tool | Purpose |
|------|---------|
| [CMake](https://cmake.org/) | C++ build system |
| [SWIG](https://www.swig.org/) | Python bindings |
| [scikit-build-core](https://scikit-build-core.readthedocs.io/) | Python wheel build backend |
| [cmake-openeye](https://github.com/scott-arne/cmake-openeye) | OpenEye CMake discovery and SWIG helpers |
| [vrzn](https://github.com/scott-arne/vrzn) | Version synchronization |
| [pytest](https://docs.pytest.org/) | Python tests |
| [Sphinx](https://www.sphinx-doc.org/) | Documentation |

## License

MIT
