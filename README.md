# OEFP

High-performance molecular fingerprints for the [OpenEye Toolkits](https://www.eyesopen.com/).

OEFP generates RDKit-compatible Morgan and topological Atom Pair fingerprints
from OpenEye molecules, stores them in compact C++ containers, and compares
them with fast scalar and batch kernels. It also provides raw Morgan and
topological Atom Pair descriptor rows plus schema-backed Mordred-compatible and
RDKit-compatible named descriptor surfaces. Python
bindings are built with SWIG, so `openeye.oechem` molecules pass directly into
C++ without serialization.

OEFP currently supports dense binary, sparse binary, and sparse counted
fingerprint containers; scalar comparison; query-to-batch comparison; `cdist`;
SciPy-compatible condensed `pdist`; columnar descriptor batches; and
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
| RDKit-compatible | Schema-backed named descriptor rows | 213 of RDKit's 2D descriptors reproduced natively and matched to RDKit 2026.03.3 within tolerance tiers; `SPS` and three always-zero VSA bins are excluded |

Mordred-compatible descriptors use local Mordred and RDKit source as the
reference truth. Descriptor definitions include source metadata, group labels,
serialized parameters, value type, description, and prerequisite bitmaps.
Coordinate prerequisites are declarative only: OEFP descriptor calculators do
not invoke conformer generation.

Morgan supports both ECFP-style connectivity invariants (default) and
FCFP-style pharmacophore-feature invariants via `use_features=True` (Donor,
Acceptor, Aromatic, Halogen, Basic, Acidic), matching RDKit's feature
atom-invariant generator and composing with `use_chirality`.
`morgan_fingerprint(mol, radius=2)` is ECFP4; `morgan_fingerprint(mol,
radius=2, use_features=True)` is FCFP4.

Morgan, Topological Atom Pair, and Topological Torsions outputs support
RDKit-compatible chirality encoding with `use_chirality=True`. OEFP keeps the
caller's OpenEye molecule graph as the input truth; it does not normalize
molecules to RDKit's graph model. When OpenEye and RDKit materialize a molecule
differently, such as stereo hydrogens or sanitization-specific valence rewrites,
chirality-enabled output may reflect that graph-model boundary.

Current conformance scope is otherwise explicit: Distance Atom Pair generation
raises `ValueError` or `NotImplementedError` until that path has dedicated RDKit
parity coverage. Topological Atom Pair uses only the molecular connectivity
graph; it does not require 2D coordinates.

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
