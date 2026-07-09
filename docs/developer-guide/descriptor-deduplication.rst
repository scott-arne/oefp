Descriptor Deduplication
========================

OEFP supports computing molecular descriptors from multiple independent sources
(Mordred-compatible descriptors, OpenEye molecular properties, etc.) and merging
them into a unified schema-backed descriptor table. When two sources expose
semantically identical descriptors — for example, both compute exact molecular
weight using the same atomic masses and rounding — the **descriptor
deduplication framework** resolves the collision by keeping the first occurrence
and dropping duplicates.

Canonical Identity
------------------

The deduplication mechanism uses a field called **canonical_id**. A
``canonical_id`` is a namespaced string (for example,
``"quantity:exact_molecular_weight"``) that is assigned ONLY where descriptors
run **the same computation** and produce **provably identical values** for every
molecule.

.. important::

   Canonical identity denotes **computational equivalence**, not conceptual
   similarity. Two descriptors are tagged with the same ``canonical_id`` when
   they:

   - Invoke the same algorithm (e.g., exact molecular weight via atomic mass
     summation).
   - Use identical numeric constants, rounding rules, and precision.
   - Produce bit-for-bit or floating-point-identical values.

   Descriptors that calculate conceptually similar but differently computed
   quantities (for example, Wildman-Crippen SLogP vs OpenEye XLogP, or
   different SMARTS-based H-bond donor counts) are NOT deduplicated — they each
   receive an empty ``canonical_id`` (``""``) marking them as unique.

   Removing merely **correlated** descriptors is the user's responsibility, not
   the calculator's.

An empty ``canonical_id`` (``""``) means the descriptor has no known
cross-source equivalent and will never be deduplicated.

Resolution Order and First-Wins
--------------------------------

When constructing a ``DescriptorCalculator``, sources are registered in order.
The calculator builds the merged schema by processing each source sequentially
and checking for ``canonical_id`` collisions:

- The **first** source that exposes a descriptor with a given
  ``canonical_id`` wins; that descriptor is kept in the final schema.
- Later sources that expose descriptors with the same ``canonical_id`` have
  those duplicates silently dropped.
- Descriptors with empty ``canonical_id`` (unique computations) are never
  dropped.

The resolution is performed once at construction time and does not change for
the calculator's lifetime. When computing descriptor values, missing values
remain missing: if the first source for a particular ``canonical_id`` does not
produce a value (for example, a 3D descriptor on a molecule without
coordinates), the output row contains ``None`` for that descriptor, even if a
later source could have computed it.

Descriptor Sources and Calculator
----------------------------------

The descriptor calculator is built from one or more **descriptor sources**:

- **DescriptorSource (abstract)**: Base type for any source of descriptors.
- **MordredDescriptorSource**: Exposes the full Mordred 1.2.0 schema with
  curated ``canonical_id`` tags for descriptors known to be identical across
  sources. OEFP implements a subset of Mordred descriptors; unimplemented or
  unavailable values are left missing.
- **OpenEyePropertyDescriptorSource**: Exposes OpenEye molecular-property
  descriptors such as exact molecular weight, heavy atom count, and predicted
  logP (XLogP). Several OpenEye descriptors share ``canonical_id`` tags with
  their Mordred equivalents where computation is provably identical.
- **RDKitDescriptorSource**: Exposes 213 RDKit 2D descriptors natively
  reproduced in OEFP and matched to RDKit 2026.03.3 within per-descriptor
  tolerance tiers. RDKit is used only as a test-time conformance oracle, not a
  runtime dependency.

**DescriptorCalculator** is constructed from a sequence of sources. Each source
can be registered:

- **Bare**: All descriptors from that source are candidates for the merged
  schema.
- **With a name selection**: Only the named descriptors from that source are
  considered.

Example:

.. code-block:: python

   import oefp

   # Register both Mordred and OpenEye sources (no name selection).
   calc = oefp.DescriptorCalculator([
       oefp.MordredDescriptorSource(),
       oefp.OpenEyePropertyDescriptorSource(),
   ])

   # Inspect the merged schema.
   print(len(calc.schema.names))  # Total number of unique descriptors
   print("MW" in calc.schema.names)

   # Compute one row.
   from openeye import oechem
   mol = oechem.OEGraphMol()
   oechem.OESmilesToMol(mol, "CCO")
   row = calc.compute(mol)
   print(row["MW"])

   # Compute a batch.
   mols = [mol]  # (Typically many molecules)
   batch = calc.calculate_batch(mols)

The calculator exposes:

- **schema**: A ``DescriptorSchema`` describing the merged, deduplicated
  descriptor columns.
- **compute(mol)**: Compute one schema-backed ``DescriptorSet`` for a single
  molecule.
- **calculate_batch(molecules)**: Compute a schema-backed ``DescriptorBatch``
  for an iterable of molecules. This method releases the GIL and computes
  descriptor rows in parallel using OEFP's native thread pool.

Shared Compute Context and Pruning
----------------------------------

Both ``compute`` and ``calculate_batch`` share a per-molecule **compute
context** while producing a row. The context memoizes molecule-level
intermediates — ring-perceived working molecule, heavy-atom graph and distance
matrix, and Gasteiger/Crippen atom contributions — so each is computed **once**
per molecule and reused wherever it is needed rather than recomputed. The
ring-perceived molecule is shared across all sources (both Mordred and the
OpenEye property source consume it); the other intermediates are Mordred-specific
and are reused across Mordred's descriptor groups.

On top of the shared context, the calculator **prunes** computation to the
columns that actually survive selection and deduplication. When a source is
registered with a name selection, or loses columns to a first-wins
``canonical_id`` collision, the calculator asks that source to compute only the
surviving columns. A descriptor group whose emitted columns are all dropped is
skipped — unless a surviving group depends on its intermediate results, in which
case it still runs (populating that shared intermediate) but emits none of its
own columns. This dependency closure is what preserves value-invariance under
pruning.

.. important::

   Pruning is **strictly subtractive and value-invariant**. Pruning changes
   *which* descriptor groups run, never *how* a surviving column is computed. A
   pruned calculator's value for any requested column is byte-identical to the
   value the same calculator would produce with nothing pruned — the shared
   intermediates and per-column algorithms are identical either way. Pruning is
   therefore a pure performance optimization: it can only make computation
   faster, never change a result.

The shared context and pruning are internal C++ optimizations of the descriptor
compute path. There is **no Python API change**: ``compute`` and
``calculate_batch`` accept the same arguments and return the same schema-backed
results as before — only faster.

RDKit Descriptor Source
------------------------

The **RDKitDescriptorSource** natively reproduces 213 descriptors from RDKit's
2D descriptor surface, matched to RDKit 2026.03.3 as a test-time conformance
oracle. RDKit is **not** a runtime dependency; all computation is performed
natively in OEFP, and RDKit is used only during testing to verify that the
native implementation produces values within acceptable tolerances.

Conformance Testing and Tolerance Tiers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The RDKit descriptor test suite compares native OEFP values against a committed
RDKit 2026.03.3 reference fixture on a fixed molecule panel. Each descriptor is
assigned one of three **tolerance tiers** that reflect the numerical stability
of its computation:

- **exact** (1e-8): Integer counts and exact-arithmetic quantities where any
  deviation is a bug.
- **tight** (1e-4): Well-conditioned floating-point descriptors such as
  connectivity indices and information content measures, where accumulation
  order or model differences should have minimal impact.
- **loose** (1e-2): Accumulation-order-sensitive or model-dependent descriptors
  such as VSA bins, BCUT2D eigenvalues, and the composite ``qed`` metric,
  where slight numerical differences are expected.

These tiers are enforced by ``tests/python/test_rdkit_descriptors.py`` via the
``TIER_TOLERANCE`` mapping.

Deduplication and Coexistence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RDKit and Mordred columns that are **conceptually similar** but computed
differently are **both present** by design, each with an empty ``canonical_id``.
A descriptor receives a non-empty ``canonical_id`` ONLY where the RDKit
computation is provably identical-by-construction to another source's column,
verified by the canonical-identity audit
(``tests/python/test_canonical_identity_audit.py``).

This is the same computational-equivalence-not-conceptual-similarity rule that
applies to Mordred and OpenEye descriptors: deduplication removes only
identical computations, not correlated or similar-but-different quantities.

.. note::

   The canonical-identity audit compares per-source raw outputs for every
   tagged ``canonical_id`` pair and asserts exact equality (``==``). Because
   tagged descriptors are identical by construction (shared computation), any
   numerical difference is treated as a failed identity, not tolerated.

Excluded Always-Zero Bins
^^^^^^^^^^^^^^^^^^^^^^^^^^

RDKit's descriptor list includes three VSA bins that are structurally always
zero for any molecule: ``SMR_VSA8``, ``SlogP_VSA9``, and ``EState_VSA11``.
These bins correspond to fixed VSA boundary ranges that receive no atomic
contribution from any element or hybridization state. Rather than ship them as
constant-zero noise, OEFP **excludes** them entirely from the schema. Together
with ``SPS`` (excluded for a different reason, below), this leaves the RDKit
descriptor schema at 213 columns, not 217.

.. important::

   Excluded descriptors are **not in the schema** and are never emitted. There
   are **no deferred** RDKit descriptors: every one of the 213 schema columns
   is computed natively.

Excluded: SPS (SpacialScore)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All 213 schema descriptors are computed natively; there are **no deferred**
RDKit descriptors. One further RDKit descriptor, **SPS** (``SpacialScore``), is
**excluded** from the schema (like the always-zero VSA bins above) rather than
shipped with known-wrong values.

``SPS`` sums a per-heavy-atom score whose terms include a non-aromatic-ring term
and a potential-stereocenter term. OEFP implements a native, RDKit-faithful
potential-stereogenicity model for the stereo term (retained in
``src/rdkit_stereogenicity.{h,cpp}`` and unit-tested byte-identical to RDKit's
``findPotentialStereo``/legacy ``findPotentialStereoBonds`` across the tested
panel). The blocker is the **ring term**: it depends on RDKit's aromaticity
perception, which differs from OpenEye's on some conjugated ring systems — e.g.
certain porphyrin-like macrocycles RDKit treats as non-aromatic (ring term 2)
but OpenEye treats as aromatic (ring term 1), doubling their contribution. A
large-scale SureChEMBL value sweep found ``SPS`` diverging from RDKit on
~0.15% of molecules, driven by this aromaticity-model difference (plus radicals
and organometallic/coordination compounds). That difference cannot be
reproduced without porting RDKit's aromaticity model, and cannot be detected at
runtime without RDKit, so ``SPS`` is excluded rather than shipped approximate.

.. note::

   **Aromaticity-model limitation (general).** The RDKit descriptor source uses
   OpenEye's aromaticity perception, which agrees with RDKit's on ordinary
   drug-like rings but can diverge on unusual conjugated macrocycles. Any
   aromaticity-dependent RDKit descriptor may therefore differ from RDKit on
   such rare inputs. This is a known, pre-existing limitation of the native
   reproduction, surfaced by the SPS value sweep.

.. note::

   **Gasteiger-dependent descriptors now computed** (20 descriptors) — The
   partial-charge extrema (4), PEOE_VSA bins (14), and BCUT2D_CHG eigenvalues
   (2) that were previously deferred have been **re-enabled** following a
   native RDKit-Gasteiger port. The cumulene Gasteiger divergence that blocked
   these descriptors was resolved by implementing RDKit's atom-typing rules
   (RDKit types cumulene atoms as ``sp``). The Mordred source continues to use
   Mordred 1.2.0's ``sp2`` Gasteiger charges for its own PartialCharge family.

The test ``test_full_213_surface_is_enabled_or_deferred`` asserts that the
enabled descriptors equal the full 213-descriptor schema (the deferred set is
empty), ensuring no descriptor is silently missing.

.. note::

   Missing values remain missing: if a source cannot compute a requested column
   (e.g. a Mordred descriptor whose 3D-coordinate prerequisite is unsatisfied on
   a no-conformer molecule), the calculator emits ``None`` for that column even
   if a later source could compute it.

Within-Source Naming Smell
---------------------------

Cross-source name differences are expected and are precisely what
``canonical_id`` reconciles. For example, Mordred's ``MW`` and OpenEye's
``MolecularWeight`` share the ``canonical_id``
``"quantity:exact_molecular_weight"`` because they compute the same value; the
calculator drops one of them, and users see a single column.

However, if a **single source** exposes two descriptor names that invoke the
same implementation (for example, both ``"HBA"`` and ``"NumHBondAcceptors"``
map to the identical SMARTS-based function), that is a smell to question.
Either:

- The two names genuinely compute different values (even if subtly), in which
  case they should have distinct ``canonical_id`` tags (or empty
  ``canonical_id`` if untagged).
- One is a true alias, in which case the source should expose only one
  canonical name or document that both names exist for backward compatibility.

Within-source duplication is rare in OEFP's curated sources but may occur in
user-defined descriptor sources. The calculator cannot detect or fix this
automatically; it is the source implementer's responsibility.

.. _drift-rule:

Drift Rule and Verification
----------------------------

.. warning::

   **Changing any descriptor's implementation** (or a shared molecular-property
   helper that feeds multiple descriptors) requires **re-verifying every
   canonical_id it participates in** still denotes identical values.

   After any descriptor implementation change:

   1. Run the canonical-identity audit test:

      .. code-block:: bash

         PYTHONPATH=python python -m pytest tests/python/test_canonical_identity_audit.py -v

      This test compares per-source raw outputs for every tagged
      ``canonical_id`` pair on a fixed molecule panel and asserts
      exact equality (``==``): because tagged descriptors are identical by
      construction (shared computation), any numerical difference is treated
      as a failed identity, not tolerated.

   2. When ``canonical_id`` tags change (additions, removals, or name changes),
      run the SureChEMBL identity sweep to verify the new tags at scale:

      .. code-block:: bash

         OEFP_SURECHEMBL_PARQUET=/path/to/compounds.parquet OEFP_SURECHEMBL_MISSED=1 \
         PYTHONPATH=python python -m pytest tests/python/surechembl_identity_sweep.py -m surechembl -q -s

      The sweep compares tagged descriptor pairs across ~50k diverse molecules
      and always asserts there are no divergences (misassigned identity). The
      ``OEFP_SURECHEMBL_MISSED=1`` flag shown above additionally reports
      potential missed identities (untagged pairs that are numerically
      identical across the entire sample) as advisory candidates (printed with
      ``-s``); it does not fail the run.

   These two-tier checks (panel audit + large-scale sweep) together reduce the
   risk of incorrect ``canonical_id`` assignment breaking production
   deduplication.

Examples of drift scenarios:

- Upgrading the underlying OpenEye toolkit to a version with a revised
  molecular-weight calculation (e.g., changed isotope masses).
- Refactoring a shared molecular-property helper that feeds multiple descriptors.
- Adding or removing rounding in a descriptor implementation.

Any of these changes can silently invalidate existing ``canonical_id``
assignments. Always re-verify after making such changes.

Persistence Compatibility Note
-------------------------------

.. important::

   Folding ``canonical_id`` into the schema-id hash **intentionally changed
   every descriptor schema_id**. Descriptor Arrow/Parquet files written by an
   OEFP version *before* this change store the old ``schema_id`` and will fail
   ``schema_from_metadata`` validation ("schema id does not match metadata")
   when read back after upgrading.

   This is an **accepted breaking change** for a pre-1.0 release with no
   on-disk backward-compatibility guarantee. Regenerate persisted descriptor
   data after upgrading to a version that includes the canonical-identity
   framework.

Additional Resources
--------------------

- :doc:`../getting-started/quickstart` - General descriptor calculation examples
- :doc:`cicd` - SureChEMBL identity sweep and other CI checks
- Source code: ``python/oefp/api.py`` (DescriptorCalculator),
  ``include/oefp/descriptor_source.h``, ``include/oefp/descriptor_calculator.h``
