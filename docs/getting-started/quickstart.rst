Quick Start
===========

This guide shows how to generate fingerprints and compare them in Python and
C++.

Python Quick Start
------------------

Generate a Fingerprint
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from openeye import oechem
   import oefp

   mol = oechem.OEGraphMol()
   oechem.OESmilesToMol(mol, "CC(=O)OC1=CC=CC=C1C(=O)O")  # aspirin

   fp = oefp.morgan_fingerprint(mol, radius=2, num_bits=2048)

   print(fp.num_bits)
   print(fp.popcount)
   print(fp.words[:4])

Use Reusable Generators
^^^^^^^^^^^^^^^^^^^^^^^

Use generator objects when applying the same options to many molecules.

.. code-block:: python

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

   print(distances)

Compare Fingerprints
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from openeye import oechem
   import oefp

   def mol_from_smiles(smiles: str) -> oechem.OEGraphMol:
       mol = oechem.OEGraphMol()
       oechem.OESmilesToMol(mol, smiles)
       return mol

   query = oefp.morgan_fingerprint(mol_from_smiles("c1ccccc1"))
   library = [
       oefp.morgan_fingerprint(mol_from_smiles("c1ccc(O)cc1")),
       oefp.morgan_fingerprint(mol_from_smiles("CC(=O)O")),
   ]

   batch = oefp.OEFPBatch.from_fingerprints(library)
   scores = oefp.compare(query, batch, oefp.Metric.tanimoto())

   print(scores)

Use Counted and Sparse Outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from openeye import oechem
   import oefp

   mol = oechem.OEGraphMol()
   oechem.OESmilesToMol(mol, "c1ccccc1O")

   folded_count = oefp.morgan_count_fingerprint(mol)
   sparse_binary = oefp.morgan_sparse_fingerprint(mol)
   atom_pair_count = oefp.atom_pair_sparse_count_fingerprint(mol)

   print(folded_count.indices[:5])
   print(folded_count.counts[:5])
   print(sparse_binary.indices[:5])
   print(atom_pair_count.total_count)

Compare Raw Descriptors
^^^^^^^^^^^^^^^^^^^^^^^

Raw descriptors keep unfurled feature keys and counts instead of folding them
into a fixed-length fingerprint. Use ``morgan_descriptors()`` for raw Morgan
environment identifiers, ``atom_pair_descriptors()`` for raw Atom Pair
features, or ``mordred_descriptors()`` for the supported Mordred-compatible
count subset. Descriptor sets use the same ``compare``, ``cdist``, and
``pdist`` functions as fingerprints.

.. code-block:: python

   from openeye import oechem
   import oefp

   def mol_from_smiles(smiles: str) -> oechem.OEGraphMol:
       mol = oechem.OEGraphMol()
       oechem.OESmilesToMol(mol, smiles)
       return mol

   query = oefp.atom_pair_descriptors(mol_from_smiles("c1ccncc1"))
   library = [
       oefp.atom_pair_descriptors(mol_from_smiles("c1ccccc1")),
       oefp.atom_pair_descriptors(mol_from_smiles("c1ccc(O)cc1")),
       oefp.atom_pair_descriptors(mol_from_smiles("CC(C)(C)Cl")),
   ]

   batch = oefp.DescriptorBatch.from_descriptors(library)

   scores = oefp.compare(
       query,
       batch,
       oefp.Metric.tanimoto(),
       descriptor_mode="presence",
   )
   distances = oefp.cdist(
       oefp.DescriptorBatch.from_descriptors([query]),
       batch,
       oefp.Metric.jaccard(),
       descriptor_mode="count_overlap",
   )
   pairwise = oefp.pdist(
       batch,
       oefp.Metric.tanimoto(),
       descriptor_mode="exact_count",
   )

   print(query.string_keys[:5])
   print(query.counts[:5])
   print(scores)
   print(distances)
   print(pairwise)

Descriptor comparison modes control how counts are interpreted:

- ``count_overlap`` compares count-aware overlap with per-key minimum and
  maximum counts.
- ``presence`` ignores counts and compares only whether a key is present.
- ``exact_count`` treats a key as shared only when both molecules have the same
  count for that key.

Inspect Morgan Bit Mappings
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mapping APIs return RDKit-style ``bitInfoMap`` entries for Morgan fingerprints.
Each entry is ``(center_atom_id, radius)``.

.. code-block:: python

   from openeye import oechem
   import oefp

   mol = oechem.OEGraphMol()
   oechem.OESmilesToMol(mol, "CCO")

   result = oefp.morgan_fingerprint_with_mapping(mol)
   bit_info = result.mapping.bit_info()

   print(result.fingerprint.popcount)
   print(bit_info)

OpenEye Fingerprint Interop
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from openeye import oechem, oegraphsim
   import oefp

   mol = oechem.OEGraphMol()
   oechem.OESmilesToMol(mol, "CCO")

   oe_fp = oegraphsim.OEFingerPrint()
   oegraphsim.OEMakeCircularFP(oe_fp, mol)

   fp = oefp.from_openeye_fingerprint(oe_fp)
   round_tripped = oefp.to_openeye_fingerprint(fp)

   print(fp.popcount)
   print(oegraphsim.OETanimoto(oe_fp, round_tripped))

.. _api-comparison:

API Comparison
--------------

RDKit, OEFP, and OEGraphSim solve overlapping but different problems. RDKit
owns the reference generator API for RDKit molecules. OEFP provides
RDKit-compatible Morgan and Atom Pair fingerprints for OpenEye molecules.
OEGraphSim provides native OpenEye fingerprint generation and similarity.

.. list-table::
   :widths: 24 25 26 25
   :header-rows: 1

   * - Task
     - RDKit
     - OEFP
     - OEGraphSim
   * - Molecule input
     - ``rdkit.Chem.Mol``
     - ``openeye.oechem.OEMolBase``
     - ``openeye.oechem.OEMolBase``
   * - Morgan or circular binary fingerprint
     - ``GetMorganGenerator(...).GetFingerprint(mol)``
     - ``oefp.morgan_fingerprint(mol, ...)``
     - ``OEMakeCircularFP(fp, mol, ...)``
   * - Atom Pair fingerprint
     - ``GetAtomPairGenerator(...).GetFingerprint(mol)``
     - ``oefp.atom_pair_fingerprint(mol, ...)``
     - No RDKit-style Atom Pair generator surface
   * - Count and sparse Morgan outputs
     - ``GetCountFingerprint()``, ``GetSparseFingerprint()``,
       ``GetSparseCountFingerprint()``
     - ``morgan_count_fingerprint()``, ``morgan_sparse_fingerprint()``,
       ``morgan_sparse_count_fingerprint()``
     - Dense binary ``OEFingerPrint`` workflow
   * - Raw counted descriptors
     - Sparse count fingerprints expose raw identifiers for supported
       generators
     - ``morgan_descriptors()``, ``atom_pair_descriptors()``, and
       ``mordred_descriptors()`` with ``DescriptorBatch``
     - No raw descriptor batch comparison surface
   * - Morgan bit provenance
     - ``AdditionalOutput`` with ``GetBitInfoMap()``
     - ``morgan_*_with_mapping(...).mapping.bit_info()``
     - No RDKit-style bit-info map
   * - Pairwise similarity
     - ``DataStructs.TanimotoSimilarity()`` or
       ``BulkTanimotoSimilarity()``
     - ``oefp.compare()``, ``oefp.cdist()``, ``oefp.pdist()``
     - ``OETanimoto()``
   * - Batch storage
     - Python lists of RDKit fingerprint objects
     - ``OEFPBatch``, ``OEFPSparseBatch``, ``OEFPCountBatch``
     - Python lists of ``OEFingerPrint`` objects
   * - Runtime dependency
     - RDKit
     - OpenEye Toolkits; RDKit is not required at runtime
     - OpenEye Toolkits

Generate a Morgan or Circular Fingerprint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use OEFP when you need RDKit-compatible Morgan output from an OpenEye molecule.
Use OEGraphSim when you want OpenEye's native circular fingerprint.

.. code-block:: python

   # RDKit
   from rdkit import Chem
   from rdkit.Chem import rdFingerprintGenerator

   rd_mol = Chem.MolFromSmiles("CCO")
   rd_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
   rd_fp = rd_gen.GetFingerprint(rd_mol)

.. code-block:: python

   # OEFP
   from openeye import oechem
   import oefp

   oe_mol = oechem.OEGraphMol()
   oechem.OESmilesToMol(oe_mol, "CCO")
   fp = oefp.morgan_fingerprint(oe_mol, radius=2, num_bits=2048)

.. code-block:: python

   # OEGraphSim
   from openeye import oechem, oegraphsim

   oe_mol = oechem.OEGraphMol()
   oechem.OESmilesToMol(oe_mol, "CCO")

   oe_fp = oegraphsim.OEFingerPrint()
   oegraphsim.OEMakeCircularFP(oe_fp, oe_mol)

Compare Two Fingerprints
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # RDKit
   from rdkit import DataStructs

   score = DataStructs.TanimotoSimilarity(rd_fp_a, rd_fp_b)

.. code-block:: python

   # OEFP
   import oefp

   score = oefp.compare(fp_a, fp_b, oefp.Metric.tanimoto())

.. code-block:: python

   # OEGraphSim
   from openeye import oegraphsim

   score = oegraphsim.OETanimoto(oe_fp_a, oe_fp_b)

Compute Pairwise Distances
^^^^^^^^^^^^^^^^^^^^^^^^^^

OEFP returns SciPy-compatible condensed ``pdist`` output directly. RDKit and
OEGraphSim users typically build the condensed vector with an explicit loop.

.. code-block:: python

   # RDKit
   from rdkit import DataStructs

   distances = []
   for index, fp in enumerate(rd_fps):
       scores = DataStructs.BulkTanimotoSimilarity(fp, rd_fps[index + 1 :])
       distances.extend(1.0 - score for score in scores)

.. code-block:: python

   # OEFP
   import oefp

   batch = oefp.OEFPBatch.from_fingerprints(fps)
   distances = oefp.pdist(batch, oefp.Metric.jaccard())

.. code-block:: python

   # OEGraphSim
   from openeye import oegraphsim

   distances = []
   for index, fp in enumerate(oe_fps):
       for other in oe_fps[index + 1 :]:
           distances.append(1.0 - oegraphsim.OETanimoto(fp, other))

Inspect Morgan Bit Provenance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OEFP follows RDKit's Morgan bit-info shape: each bit maps to one or more
``(center_atom_id, radius)`` environments.

.. code-block:: python

   # RDKit
   from rdkit.Chem import rdFingerprintGenerator

   output = rdFingerprintGenerator.AdditionalOutput()
   output.AllocateBitInfoMap()
   rd_gen.GetFingerprint(rd_mol, additionalOutput=output)
   bit_info = output.GetBitInfoMap()

.. code-block:: python

   # OEFP
   import oefp

   result = oefp.morgan_fingerprint_with_mapping(oe_mol)
   bit_info = result.mapping.bit_info()

OEGraphSim does not expose an RDKit-style Morgan ``bitInfoMap``.

C++ Quick Start
---------------

Generate and Compare
^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   #include <oefp/oefp.h>
   #include <oechem.h>
   #include <iostream>

   int main() {
       OEChem::OEGraphMol mol_a;
       OEChem::OEGraphMol mol_b;
       OEChem::OESmilesToMol(mol_a, "c1ccccc1");
       OEChem::OESmilesToMol(mol_b, "c1ccc(O)cc1");

       OEFP::MorganOptions options;
       options.radius = 2;
       options.num_bits = 2048;

       OEFP::OEFP fp_a = OEFP::MakeMorganFingerprint(mol_a, options);
       OEFP::OEFP fp_b = OEFP::MakeMorganFingerprint(mol_b, options);

       double score = OEFP::Compare(fp_a, fp_b, OEFP::Metric::Tanimoto());
       std::cout << score << "\n";

       return 0;
   }

Batch ``pdist``
^^^^^^^^^^^^^^^

.. code-block:: cpp

   #include <oefp/oefp.h>
   #include <oechem.h>
   #include <vector>

   int main() {
       std::vector<OEFP::OEFP> fps;
       OEFP::MorganGenerator generator;

       for (const char* smiles : {"c1ccccc1", "c1ccc(O)cc1", "CC(=O)O"}) {
           OEChem::OEGraphMol mol;
           OEChem::OESmilesToMol(mol, smiles);
           fps.push_back(generator.Fingerprint(mol));
       }

       OEFP::OEFPBatch batch = OEFP::OEFPBatch::FromFingerprints(fps);
       std::vector<double> distances =
           OEFP::PDist(batch, OEFP::Metric::Jaccard());

       return distances.empty() ? 1 : 0;
   }

Supported Generator Scope
-------------------------

.. list-table::
   :widths: 30 45 25
   :header-rows: 1

   * - Family
     - Supported outputs
     - Notes
   * - Morgan
     - Folded binary, folded count, sparse binary, sparse count, raw
       descriptors
     - Bit mapping is available for all Morgan outputs
   * - Atom Pair
     - Folded binary, folded count, sparse binary, sparse count, raw
       descriptors
     - Count simulation is enabled by default for binary output
   * - OpenEye
     - Import/export of ``OEFingerPrint``
     - Numeric type metadata is preserved when available

Current Boundaries
------------------

The unsupported paths fail explicitly:

- Morgan chirality
- Atom Pair chirality
- Atom Pair 3D-distance generation

These options will remain disabled until they have dedicated RDKit parity
coverage.

Next Steps
----------

- :doc:`../python-lib/index` - Python library documentation
- :doc:`../cpp-api/library_root` - C++ API reference
