# OEFP

High-performance molecular fingerprints for the [OpenEye Toolkits](https://www.eyesopen.com/).

OEFP generates RDKit-compatible Morgan and topological Atom Pair fingerprints
from OpenEye molecules, stores them in compact C++ containers, and compares
them with fast scalar and batch kernels. It also provides raw Morgan and
topological Atom Pair descriptor rows plus a schema-backed Mordred-compatible
descriptor surface. Python
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

Mordred-compatible descriptors use local Mordred and RDKit source as the
reference truth. Descriptor definitions include source metadata, group labels,
serialized parameters, value type, description, and prerequisite bitmaps.
Coordinate prerequisites are declarative only: OEFP descriptor calculators do
not invoke conformer generation.

Morgan and Topological Atom Pair outputs support RDKit-compatible chirality
encoding with `use_chirality=True`. OEFP keeps the caller's OpenEye molecule
graph as the input truth; it does not normalize molecules to RDKit's graph
model. When OpenEye and RDKit materialize a molecule differently, such as stereo
hydrogens or sanitization-specific valence rewrites, chirality-enabled output
may reflect that graph-model boundary.

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
