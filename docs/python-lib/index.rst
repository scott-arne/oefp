Python Library (oefp)
=====================

The ``oefp`` Python package provides SWIG-generated bindings to the C++ OEFP
library. It works directly with OpenEye molecules and exposes Pythonic wrapper
objects for fingerprint storage and comparison.

Overview
--------

The main Python components are:

- :class:`OEFP` - Dense binary fingerprint
- :class:`OEFPSparse` - Sparse binary fingerprint
- :class:`OEFPCount` - Sparse counted fingerprint
- :class:`OEFPBatch` - Dense binary batch storage
- :class:`OEFPSparseBatch` - Sparse binary batch storage
- :class:`OEFPCountBatch` - Sparse counted batch storage
- :class:`FingerprintSpec` - Read-only fingerprint metadata
- :class:`Metric` - Tanimoto similarity, Jaccard distance, Dice, Cosine,
  Tversky, and Manhattan metrics
- :class:`MorganGenerator` - Reusable Morgan dense-binary generator
- :class:`AtomPairGenerator` - Reusable Atom Pair dense-binary generator

For side-by-side RDKit, OEFP, and OEGraphSim examples, see
:ref:`api-comparison`.

Installation
------------

Install via pip:

.. code-block:: bash

   pip install openeye-toolkits
   pip install oefp

Or build from source:

.. code-block:: bash

   python scripts/build_python.py --openeye-root /path/to/openeye/sdk
   pip install dist/oefp-*.whl

Basic Usage
-----------

.. code-block:: python

   from openeye import oechem
   import oefp

   mol = oechem.OEGraphMol()
   oechem.OESmilesToMol(mol, "CCO")

   fp = oefp.morgan_fingerprint(mol)
   score = oefp.compare(fp, fp, oefp.Metric.tanimoto())

   print(fp.popcount)
   print(score)

Fingerprint Containers
----------------------

.. class:: OEFP

   Dense binary fingerprint backed by native ``uint64`` words.

   .. method:: OEFP.from_on_bits(num_bits, bits, *, algorithm="manual")

      Create a dense binary fingerprint from on-bit indices.

   .. attribute:: words

      Read-only NumPy view of the native ``uint64`` word storage.

   .. attribute:: popcount

      Number of on bits.

   .. attribute:: num_bits

      Fixed fingerprint size in bits.

   .. attribute:: spec

      Read-only :class:`FingerprintSpec` metadata.

   Example::

      fp = oefp.OEFP.from_on_bits(128, [1, 5, 127])
      print(fp.words)
      print(fp.popcount)

.. class:: OEFPSparse

   Sparse binary fingerprint backed by sorted ``uint32`` identifiers.

   .. attribute:: indices

      Read-only NumPy view of sorted on-bit identifiers.

   .. attribute:: popcount

      Number of on-bit entries.

   .. attribute:: num_bits

      Sparse identifier domain size.

   .. attribute:: spec

      Read-only :class:`FingerprintSpec` metadata.

.. class:: OEFPCount

   Sparse counted fingerprint backed by sorted ``uint32`` identifiers and
   parallel ``uint32`` counts.

   .. attribute:: indices

      Read-only NumPy view of sorted count identifiers.

   .. attribute:: counts

      Read-only NumPy view of counts parallel to ``indices``.

   .. attribute:: nonzero_count

      Number of nonzero count entries.

   .. attribute:: total_count

      Sum of all sparse counts.

   .. attribute:: spec

      Read-only :class:`FingerprintSpec` metadata.

Batch Containers
----------------

.. class:: OEFPBatch

   Contiguous row-major dense binary batch.

   .. method:: OEFPBatch.from_fingerprints(fingerprints)

      Create a batch from ``OEFP`` objects with matching specifications.

   .. attribute:: words

      Read-only 2D NumPy view of row-major ``uint64`` words.

   .. attribute:: popcounts

      Read-only NumPy view of one cached popcount per row.

   .. attribute:: size

      Number of fingerprint rows.

   .. attribute:: spec

      Read-only :class:`FingerprintSpec` metadata shared by all rows.

   Example::

      fps = [oefp.morgan_fingerprint(mol) for mol in mols]
      batch = oefp.OEFPBatch.from_fingerprints(fps)
      values = oefp.pdist(batch, oefp.Metric.jaccard())

.. class:: OEFPSparseBatch

   Contiguous sparse binary batch using flattened indices and row offsets.

.. class:: OEFPCountBatch

   Contiguous sparse counted batch using flattened indices, counts, and row
   offsets.

.. class:: FingerprintSpec

   Immutable metadata describing a fingerprint's size, value type, source, and
   generator parameters.

   Example::

      fp = oefp.morgan_fingerprint(mol)
      print(fp.spec.num_bits)
      print(fp.spec.source_type)

Metrics and Comparison
----------------------

.. class:: Metric

   Metric configuration used by scalar and batch comparison functions.

   .. method:: Metric.tanimoto(*, mode="similarity")
   .. method:: Metric.jaccard(*, mode="distance")
   .. method:: Metric.dice(*, mode="similarity")
   .. method:: Metric.cosine(*, mode="similarity")
   .. method:: Metric.tversky(alpha, beta, *, mode="similarity")
   .. method:: Metric.manhattan(*, mode="distance")

      Create a metric. Tanimoto is a similarity metric. Jaccard is the
      corresponding distance metric. Dice, Cosine, and Tversky accept either
      ``"similarity"`` or ``"distance"``.

.. function:: compare(a, b, metric, *, num_threads=0, chunk_size=256)

   Compare two fingerprints or one query fingerprint against a matching batch.

   Example::

      score = oefp.compare(fp_a, fp_b, oefp.Metric.tanimoto())
      scores = oefp.compare(fp_a, batch, oefp.Metric.tanimoto())

.. function:: cdist(a, b, metric, *, num_threads=0, chunk_size=256)

   Return row-major cross-comparison values for two matching batch containers.

.. function:: pdist(batch, metric, *, num_threads=0, chunk_size=256)

   Return SciPy-compatible condensed pairwise values for one batch.

Morgan Fingerprints
-------------------

.. class:: MorganGenerator(...)

   Reusable generator for folded dense binary Morgan fingerprints.

   Constructor options:

   - ``radius=2``
   - ``num_bits=2048``
   - ``use_chirality=False``
   - ``use_bond_types=True``
   - ``only_nonzero_invariants=False``
   - ``include_ring_membership=True``
   - ``include_redundant_environments=False``
   - ``count_simulation=False``
   - ``count_bounds=None``

   .. method:: fingerprint(mol)

      Generate a folded dense binary Morgan fingerprint.

.. function:: morgan_fingerprint(mol, *, radius=2, num_bits=2048, ...)

   Generate an RDKit-compatible folded binary Morgan fingerprint.

.. function:: morgan_count_fingerprint(mol, *, radius=2, num_bits=2048, ...)

   Generate a folded count Morgan fingerprint.

.. function:: morgan_sparse_fingerprint(mol, *, radius=2, ...)

   Generate a sparse binary Morgan fingerprint with raw identifiers.

.. function:: morgan_sparse_count_fingerprint(mol, *, radius=2, ...)

   Generate a sparse count Morgan fingerprint with raw identifiers.

.. function:: morgan_fingerprint_with_mapping(mol, *, ...)
              morgan_count_fingerprint_with_mapping(mol, *, ...)
              morgan_sparse_fingerprint_with_mapping(mol, *, ...)
              morgan_sparse_count_fingerprint_with_mapping(mol, *, ...)

   Generate Morgan fingerprints with RDKit-style bit environment mappings.

Atom Pair Fingerprints
----------------------

.. class:: AtomPairGenerator(...)

   Reusable generator for folded dense binary Atom Pair fingerprints.

   Constructor options:

   - ``min_distance=1``
   - ``max_distance=30``
   - ``num_bits=2048``
   - ``use_chirality=False``
   - ``use_2d=True``
   - ``count_simulation=True``
   - ``count_bounds=None``

   .. method:: fingerprint(mol)

      Generate a folded dense binary Atom Pair fingerprint.

.. function:: atom_pair_fingerprint(mol, *, min_distance=1, max_distance=30, num_bits=2048, ...)

   Generate an RDKit-compatible folded binary Atom Pair fingerprint.

.. function:: atom_pair_count_fingerprint(mol, *, ...)

   Generate a folded count Atom Pair fingerprint.

.. function:: atom_pair_sparse_fingerprint(mol, *, ...)

   Generate a sparse binary Atom Pair fingerprint.

.. function:: atom_pair_sparse_count_fingerprint(mol, *, ...)

   Generate a sparse count Atom Pair fingerprint.

Mapping Results
---------------

.. class:: OEFPMappingSet

   Wrapper for fingerprint bit environment mappings.

   .. method:: bit_ids(row=0)

      Return mapped bit ids for a fingerprint row.

   .. method:: atoms_for_bit(bit_id, row=0)

      Return center atom ids recorded for one bit.

   .. method:: environments_for_bit(bit_id, row=0)

      Return ``(atom_id, radius)`` environments recorded for one bit.

   .. method:: bit_info(row=0)

      Return an RDKit-style bit-info mapping.

OpenEye Interop
---------------

.. function:: from_openeye_fingerprint(fp)

   Import an OpenEye ``OEFingerPrint`` as an ``OEFP``.

.. function:: to_openeye_fingerprint(fp)

   Export an ``OEFP`` as an OpenEye ``OEFingerPrint`` when the fingerprint
   carries OpenEye source metadata.

Conformance Notes
-----------------

OEFP's Morgan and Atom Pair generators are implemented in OEFP and tested
against RDKit for supported options. RDKit is not a runtime dependency.

Unsupported options fail explicitly:

- Morgan chirality
- Atom Pair chirality
- Atom Pair 3D-distance generation
