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
