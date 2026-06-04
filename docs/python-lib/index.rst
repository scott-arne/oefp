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
- :class:`DescriptorDefinition` - Named descriptor column metadata
- :class:`DescriptorSchema` - Ordered schema for named descriptor rows
- :class:`DescriptorSet` - Raw counted descriptors or schema-backed named rows
- :class:`DescriptorBatch` - Batch storage for descriptor rows
- :class:`FingerprintSpec` - Read-only fingerprint metadata
- :class:`DescriptorSpec` - Read-only descriptor metadata
- :class:`Metric` - Scikit-learn-style distance metrics plus Tanimoto and
  Tversky similarity metrics
- :class:`MorganGenerator` - Reusable Morgan dense-binary generator
- :class:`TopologicalAtomPairGenerator` - Reusable topological Atom Pair
  dense-binary generator
- :func:`mordred_schema` and :func:`mordred_descriptors` -
  Mordred-compatible schema-backed descriptors

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

Descriptor Containers
---------------------

OEFP has two descriptor row shapes:

- **Raw counted descriptors** store feature keys and counts. Morgan and Atom
  Pair descriptor factories use this shape, and the rows can be compared with
  ``compare()``, ``cdist()``, and ``pdist()``.
- **Schema-backed descriptors** store named scalar values against an ordered
  :class:`DescriptorSchema`. Mordred-compatible descriptors use this shape.
  Missing values are explicit ``None`` values in Python.

Schema-backed rows and batches are tabular data. They support named access,
column extraction, subsetting, and Arrow/Parquet interchange, but they are not
accepted by the descriptor comparison kernels.

.. data:: DESCRIPTOR_PREREQUISITE_NONE
          DESCRIPTOR_PREREQUISITE_GRAPH
          DESCRIPTOR_PREREQUISITE_COORDINATES_2D
          DESCRIPTOR_PREREQUISITE_COORDINATES_3D
          DESCRIPTOR_PREREQUISITE_ALL
          TOPOLOGICAL_ATOM_PAIR_PREREQUISITES
          DISTANCE_ATOM_PAIR_PREREQUISITES

   Integer bitmap constants used by :class:`DescriptorDefinition`
   ``prerequisites`` metadata.

.. function:: descriptor_missing_prerequisites(required, available)

   Return prerequisite bits that are required by a descriptor but absent from
   an input.

.. function:: descriptor_prerequisites_satisfied(required, available)

   Return ``True`` when all required prerequisite bits are present.

.. class:: DescriptorDefinition

   Immutable metadata for one schema-backed descriptor column.

   Constructor fields:

   - ``name``
   - ``value_type``: ``"float"``, ``"int"``, ``"bool"``, or ``"string"``
   - ``group``
   - ``source_name``
   - ``source_type``
   - ``source_version``
   - ``parameters``
   - ``units``
   - ``description``
   - ``prerequisites``

   ``prerequisites`` is a ``uint32`` bitmap. Descriptor factories use this
   metadata to leave values missing when the input molecule does not satisfy a
   descriptor's requirements. The bitmap is also serialized in Arrow and
   Parquet schema metadata.

.. class:: DescriptorSchema

   Ordered collection of :class:`DescriptorDefinition` objects.

   .. attribute:: definitions

      Tuple of descriptor definitions in schema order.

   .. attribute:: names

      Tuple of descriptor names in schema order.

   .. attribute:: schema_id

      Stable identifier derived from descriptor definitions and prerequisite
      metadata.

   .. method:: index(name)

      Return the integer position for a descriptor name.

   .. method:: group(group)

      Return positions that belong to a descriptor group.

   .. method:: subset(names)

      Return a schema projected to descriptor names in selection order.

   Example::

      schema = oefp.DescriptorSchema(
          [
              oefp.DescriptorDefinition("MW", "float"),
              oefp.DescriptorDefinition(
                  "GeomDiameter",
                  "float",
                  prerequisites=oefp.DESCRIPTOR_PREREQUISITE_COORDINATES_3D,
              ),
          ]
      )

.. class:: DescriptorSet

   Python wrapper for legacy raw counted descriptor rows or schema-backed
   named descriptor rows.

   For schema-backed rows, construct with ``DescriptorSet(schema, values)``.
   Values are addressed by descriptor name and may be ``None`` when a
   descriptor is unsupported or unavailable.

   .. method:: DescriptorSet(schema, values, *, row_id="")

      Create a schema-backed descriptor row from a
      :class:`DescriptorSchema` and a mapping of descriptor names to values.

   .. method:: subset(names)

      Return a schema-backed row projected to named columns.

   .. attribute:: schema

      Descriptor schema for schema-backed rows.

   .. attribute:: row_id

      Optional row identifier for schema-backed rows.

   Example::

      schema = oefp.DescriptorSchema(
          [
              oefp.DescriptorDefinition("MW", "float"),
              oefp.DescriptorDefinition("nAtom", "int"),
          ]
      )
      row = oefp.DescriptorSet(schema, {"MW": 46.069, "nAtom": 9})
      print(row["MW"])
      print(row.subset(["nAtom"]).schema.names)

   Raw counted descriptor rows store one key vector for their active value
   type and a parallel ``uint32`` count vector.

   .. method:: DescriptorSet.from_strings(keys, counts=None, *, spec=None, source_name="OEFP", source_type="manual", source_version="", parameters="")

      Create counted string-key descriptors.

   .. method:: DescriptorSet.from_integers(keys, counts=None, *, spec=None, source_name="OEFP", source_type="manual", source_version="", parameters="")

      Create counted integer-key descriptors.

   .. method:: DescriptorSet.from_floats(keys, counts=None, *, spec=None, source_name="OEFP", source_type="manual", source_version="", parameters="")

      Create counted float-key descriptors.

   .. attribute:: value_type

      Descriptor key type: ``"string"``, ``"integer"``, or ``"float"``.

   .. attribute:: string_keys
                  integer_keys
                  float_keys

      Canonical sorted keys for the active descriptor type. Non-active key
      attributes return an empty tuple.

   .. attribute:: counts

      Read-only NumPy view of counts parallel to the active keys.

   .. attribute:: total_count

      Sum of all descriptor counts.

   .. attribute:: spec

      Read-only :class:`DescriptorSpec` metadata.

   Raw counted example::

      desc = oefp.DescriptorSet.from_strings(
          ["aromatic_N", "aromatic_C", "aromatic_C"],
          source_type="example",
      )
      print(desc.string_keys)
      print(desc.counts)

.. class:: DescriptorBatch

   Batch storage for raw counted descriptor rows or schema-backed named rows.

   .. method:: DescriptorBatch.from_descriptors(descriptors)

      Create a batch from compatible :class:`DescriptorSet` objects.

   For schema-backed rows, all rows must have the same
   :class:`DescriptorSchema`.

   .. attribute:: schema

      Shared schema for schema-backed batches.

   .. attribute:: row_ids

      Tuple of schema-backed row identifiers.

   .. method:: float_column(name)

      Return a floating-point descriptor column. Missing values are returned
      as ``numpy.nan``.

   .. method:: int_column(name)

      Return an integer descriptor column. Columns with missing values use an
      object dtype so ``None`` can be preserved.

   .. method:: bool_column(name)

      Return a boolean descriptor column. Columns with missing values use an
      object dtype so ``None`` can be preserved.

   .. method:: string_column(name)

      Return a tuple of string or ``None`` values.

   .. method:: column_validity(name)

      Return a boolean NumPy mask where ``True`` marks present values.

   .. method:: subset(names)

      Return a schema-backed batch projected to named columns.

   .. method:: to_arrow()

      Convert a schema-backed descriptor batch to a ``pyarrow.Table`` with
      OEFP schema metadata.

   .. method:: DescriptorBatch.from_arrow(table)

      Reconstruct a schema-backed batch from a ``pyarrow.Table`` produced by
      :meth:`to_arrow`.

   .. method:: write_parquet(path)

      Write a schema-backed descriptor batch to Parquet.

   .. method:: DescriptorBatch.read_parquet(path)

      Read a schema-backed descriptor batch from Parquet.

   Schema-backed example::

      rows = [
          oefp.mordred_descriptors(mol_a),
          oefp.mordred_descriptors(mol_b),
      ]
      batch = oefp.DescriptorBatch.from_descriptors(rows)
      print(batch.float_column("MW"))
      print(batch.column_validity("GeomDiameter"))

   Raw counted batches use flattened keys, counts, and row offsets. All rows
   must have matching :class:`DescriptorSpec` metadata.

   .. attribute:: string_keys
                  integer_keys
                  float_keys

      Flattened keys for the active descriptor type.

   .. attribute:: counts

      Read-only NumPy view of flattened counts.

   .. attribute:: offsets

      Read-only NumPy view of row offsets into the flattened arrays.

   .. attribute:: size

      Number of descriptor rows.

   .. attribute:: spec

      Shared :class:`DescriptorSpec` metadata.

.. class:: DescriptorSpec

   Immutable metadata describing descriptor key type, source, source version,
   and generator parameters.

Metrics and Comparison
----------------------

.. class:: Metric

   Metric configuration used by scalar and batch comparison functions.

   .. method:: Metric.euclidean()
   .. method:: Metric.manhattan()
   .. method:: Metric.chebyshev()
   .. method:: Metric.minkowski(p=2.0, weights=None)
   .. method:: Metric.standardized_euclidean(variances)
   .. method:: Metric.seuclidean(variances)
   .. method:: Metric.mahalanobis(inverse_covariance)
   .. method:: Metric.haversine()
   .. method:: Metric.hamming()
   .. method:: Metric.canberra()
   .. method:: Metric.bray_curtis()
   .. method:: Metric.jaccard()
   .. method:: Metric.matching()
   .. method:: Metric.dice()
   .. method:: Metric.kulsinski()
   .. method:: Metric.rogers_tanimoto()
   .. method:: Metric.russell_rao()
   .. method:: Metric.sokal_michener()
   .. method:: Metric.sokal_sneath()
   .. method:: Metric.tanimoto()
   .. method:: Metric.tversky(alpha, beta)

      Create a metric. The scikit-learn-style factories are distance metrics.
      ``tanimoto()`` and ``tversky()`` are boolean-space similarity metrics.
      Metric objects expose ``name``, ``type``, and ``space`` metadata.

.. function:: compare(a, b, metric, *, descriptor_mode="count_overlap", num_threads=0, chunk_size=256)

   Compare two fingerprints, two descriptor sets, or one query object against a
   matching batch. Descriptor comparison supports raw counted descriptor rows
   and batches. Schema-backed descriptor rows are tabular data and are not
   accepted by comparison kernels.

   Example::

      score = oefp.compare(fp_a, fp_b, oefp.Metric.tanimoto())
      scores = oefp.compare(fp_a, batch, oefp.Metric.tanimoto())
      descriptor_score = oefp.compare(
          desc_a,
          desc_b,
          oefp.Metric.tanimoto(),
          descriptor_mode="presence",
      )

   Descriptor comparisons accept three modes:

   - ``"count_overlap"`` compares count-aware overlap with per-key minimum and
     maximum counts.
   - ``"presence"`` ignores counts and compares only whether a key is present.
   - ``"exact_count"`` treats a key as shared only when both rows have the same
     count for that key.

.. function:: cdist(a, b, metric, *, descriptor_mode="count_overlap", num_threads=0, chunk_size=256)

   Return row-major cross-comparison values for two matching batch containers.
   Schema-backed descriptor batches are not accepted.

.. function:: pdist(batch, metric, *, descriptor_mode="count_overlap", num_threads=0, chunk_size=256)

   Return SciPy-compatible condensed pairwise values for one batch.
   Schema-backed descriptor batches are not accepted.

Morgan Fingerprints
-------------------

.. class:: MorganGenerator(...)

   Reusable generator for folded dense binary Morgan fingerprints.

   Constructor options:

   - ``radius=2``
   - ``num_bits=2048``
   - ``use_chirality=False``
   - ``use_features=False``
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

   ``use_features=True`` selects FCFP-style pharmacophore feature invariants
   (Donor, Acceptor, Aromatic, Halogen, Basic, Acidic) instead of the default
   ECFP connectivity invariants, matching RDKit's feature atom-invariant
   generator. It composes with ``use_chirality``.
   ``morgan_fingerprint(mol, radius=2)`` is ECFP4;
   ``morgan_fingerprint(mol, radius=2, use_features=True)`` is FCFP4. The same
   option is available on :class:`MorganGenerator` and the other Morgan
   fingerprint and descriptor entry points.

.. function:: morgan_count_fingerprint(mol, *, radius=2, num_bits=2048, ...)

   Generate a folded count Morgan fingerprint.

.. function:: morgan_sparse_fingerprint(mol, *, radius=2, ...)

   Generate a sparse binary Morgan fingerprint with raw identifiers.

.. function:: morgan_sparse_count_fingerprint(mol, *, radius=2, ...)

   Generate a sparse count Morgan fingerprint with raw identifiers.

.. function:: morgan_descriptors(mol, *, radius=2, use_chirality=False, use_bond_types=True, only_nonzero_invariants=False, include_ring_membership=True, include_redundant_environments=False)

   Generate raw counted Morgan descriptors as integer-key
   :class:`DescriptorSet` objects. Descriptor keys are the unfurled Morgan raw
   environment identifiers used by sparse count output. Passing
   ``use_chirality=True`` applies the same chirality encoding as Morgan
   fingerprint generation.

.. function:: morgan_fingerprint_with_mapping(mol, *, ...)
              morgan_count_fingerprint_with_mapping(mol, *, ...)
              morgan_sparse_fingerprint_with_mapping(mol, *, ...)
              morgan_sparse_count_fingerprint_with_mapping(mol, *, ...)

   Generate Morgan fingerprints with RDKit-style bit environment mappings.

Topological and Distance Atom Pair Fingerprints
-----------------------------------------------

.. class:: TopologicalAtomPairGenerator(...)

   Reusable generator for folded dense binary topological Atom Pair
   fingerprints.

   Constructor options:

   - ``min_distance=1``
   - ``max_distance=30``
   - ``num_bits=2048``
   - ``use_chirality=False``
   - ``count_simulation=True``
   - ``count_bounds=None``

   .. method:: fingerprint(mol)

      Generate a folded dense binary topological Atom Pair fingerprint.

.. class:: AtomPairGenerator(...)

   Compatibility generator with RDKit-style options. ``use_2d=True`` selects
   the topological/connectivity-distance model. ``use_2d=False`` selects the
   separate Distance Atom Pair model and fails because Distance Atom Pair is
   not implemented.

.. function:: atom_pair_fingerprint(mol, *, min_distance=1, max_distance=30, num_bits=2048, ...)
              topological_atom_pair_fingerprint(mol, *, min_distance=1, max_distance=30, num_bits=2048, ...)

   Generate a folded binary topological Atom Pair fingerprint. Topological
   Atom Pair uses graph shortest-path distances and does not require 2D or 3D
   coordinates.

.. function:: atom_pair_count_fingerprint(mol, *, ...)
              topological_atom_pair_count_fingerprint(mol, *, ...)

   Generate a folded count topological Atom Pair fingerprint.

.. function:: atom_pair_sparse_fingerprint(mol, *, ...)
              topological_atom_pair_sparse_fingerprint(mol, *, ...)

   Generate a sparse binary topological Atom Pair fingerprint.

.. function:: atom_pair_sparse_count_fingerprint(mol, *, ...)
              topological_atom_pair_sparse_count_fingerprint(mol, *, ...)

   Generate a sparse count topological Atom Pair fingerprint.

.. function:: atom_pair_descriptors(mol, *, min_distance=1, max_distance=30, use_chirality=False, use_2d=True)
              topological_atom_pair_descriptors(mol, *, min_distance=1, max_distance=30, use_chirality=False)

   Generate raw counted topological Atom Pair descriptors as string-key
   :class:`DescriptorSet` objects. Topological Atom Pair descriptors require
   only molecular graph connectivity, and support chirality-aware atom codes
   when ``use_chirality=True``.

.. function:: distance_atom_pair_fingerprint(mol, *, ...)
              distance_atom_pair_count_fingerprint(mol, *, ...)
              distance_atom_pair_sparse_fingerprint(mol, *, ...)
              distance_atom_pair_sparse_count_fingerprint(mol, *, ...)
              distance_atom_pair_descriptors(mol, *, ...)

   Reserved entry points for 3D coordinate-distance Atom Pair outputs. These
   functions require an input molecule with existing 3D coordinates and then
   fail explicitly because Distance Atom Pair parity is not implemented.

Mordred-Compatible Descriptors
------------------------------

.. function:: mordred_schema()

   Return the full generated Mordred 1.2.0 descriptor schema.

   The schema preserves Mordred calculator order and includes descriptor
   source metadata, group labels, serialized parameters, descriptions, value
   types, and prerequisite bitmaps.

.. function:: mordred_descriptors(mol)

   Generate Mordred-compatible descriptors as a schema-backed
   :class:`DescriptorSet`.

   The row uses :func:`mordred_schema`. Implemented descriptors are filled with
   typed values. Descriptors that have not been implemented, cannot be
   calculated for the molecule, or require unavailable prerequisites remain
   ``None``.

   OEFP does not generate 2D or 3D coordinates during descriptor calculation.
   Mordred descriptors whose local Mordred definitions require 3D coordinates
   declare ``DESCRIPTOR_PREREQUISITE_COORDINATES_3D``. If the input molecule is
   a 2D or no-conformer molecule, those descriptors remain missing.

   Example::

      row = oefp.mordred_descriptors(mol)
      schema = row.schema

      print(row["MW"])
      print(row["AMW"])

      requires_3d = [
          definition.name
          for definition in schema.definitions
          if (
              definition.prerequisites
              & oefp.DESCRIPTOR_PREREQUISITE_COORDINATES_3D
          )
      ]
      print(row[requires_3d[0]])  # None unless mol already has 3D coords.

   To build a tabular result across molecules, use
   :class:`DescriptorBatch`::

      batch = oefp.DescriptorBatch.from_descriptors(
          [oefp.mordred_descriptors(mol) for mol in mols]
      )
      mw = batch.float_column("MW")
      geom_present = batch.column_validity("GeomDiameter")

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

OEFP's Morgan, Atom Pair, and Topological Torsions generators are implemented in
OEFP and tested against RDKit for supported options. RDKit is not a runtime
dependency.

Morgan supports both ECFP-style connectivity invariants (default) and FCFP-style
pharmacophore-feature invariants with ``use_features=True``, matching RDKit's
feature atom-invariant generator. The feature option composes with
``use_chirality`` and applies to dense binary, sparse binary, folded count,
sparse count, bit mapping, and raw descriptor outputs.

Morgan, Atom Pair, and Topological Torsions outputs support RDKit-compatible
chirality encoding with ``use_chirality=True``. Morgan chirality is covered for
dense binary, sparse binary, folded count, sparse count, bit mapping, and raw
descriptor outputs. Atom Pair and Topological Torsions chirality are covered for
dense binary, sparse binary, folded count, sparse count, and raw descriptor
outputs. Topological Torsions chirality matches RDKit's legacy
``GetTopologicalTorsionFingerprint`` reference; the newer
``rdFingerprintGenerator`` Topological Torsions path does not encode CIP
chirality and is used only for achiral conformance.

OEFP preserves the caller's OpenEye molecule graph. It does not normalize the
molecule into RDKit's graph model before generating fingerprints. If OpenEye and
RDKit materialize a molecule differently, for example around stereo hydrogens
or sanitization-specific valence rewrites, output can differ for that reason.

The unsupported paths fail explicitly:

- Distance Atom Pair generation
- Implicit 2D or 3D coordinate generation during descriptor calculation

For Mordred-compatible descriptors, missing prerequisites are represented as
missing descriptor values rather than silently generating coordinates.
