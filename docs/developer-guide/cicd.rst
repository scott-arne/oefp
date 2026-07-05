CI and Release Builds
=====================

OEFP uses CMake for the C++ library, SWIG for Python bindings, and
scikit-build-core for wheel builds.

Local Verification
------------------

Run the core verification commands before releasing:

.. code-block:: bash

   cmake --build build-debug --target oefp_python oefp_tests
   ctest --test-dir build-debug --output-on-failure
   PYTHONPATH=python python -m pytest tests/python -q
   ruff check python tests benchmarks
   mypy python tests benchmarks
   git diff --check

RDKit Conformance and Performance
---------------------------------

RDKit is used as a conformance oracle for supported Morgan and Atom Pair
generator options. It is also used by the generation benchmark:

.. code-block:: bash

   PYTHONPATH=python python benchmarks/benchmark_rdkit_generation.py \
     --max-mols 1500 \
     --trials 7 \
     --warmup 1 \
     --pdist-size 400 \
     --generation-max-ratio 1.10 \
     --atom-pair-generation-max-ratio 1.10

The benchmark measures reusable generator calls on prebuilt molecules. It does
not include SMILES parsing time.

SureChEMBL Canonical-Identity Sweep
------------------------------------

The optional SureChEMBL sweep tests canonical-identity assignment heuristics
across ~50k diverse molecules. It requires a local parquet file and is not
part of the default test suite:

.. code-block:: bash

   OEFP_SURECHEMBL_PARQUET=/path/compounds.parquet \
   PYTHONPATH=python python -m pytest tests/python/surechembl_identity_sweep.py -m surechembl -q

The sweep compares per-source outputs for:

- **Misassigned identity**: tagged column pairs (shared ``canonical_id``) that
  diverge on any molecule. Divergences trigger an assertion failure.
- **Missed identity** (optional): untagged cross-source pairs that are
  numerically identical across the whole sample. Gated by
  ``OEFP_SURECHEMBL_MISSED=1``. Candidates are printed but do not fail.

Set ``OEFP_SURECHEMBL_SAMPLE`` to control sample size (default 50000).

This is a heuristic check, not proof. Unparseable molecules are counted and
skipped.

OECluster Guardrail
-------------------

The optional C++ benchmark compares OEFP dense comparison kernels with the
current ``oecluster`` fingerprint paths:

.. code-block:: bash

   cmake -S . -B build-bench \
     -DCMAKE_BUILD_TYPE=Release \
     -DOEFP_BUILD_TESTS=OFF \
     -DOEFP_BUILD_PYTHON=OFF \
     -DOEFP_BUILD_BENCHMARKS=ON \
     -DOEFP_OECLUSTER_SOURCE_DIR=/path/to/oecluster

   cmake --build build-bench --target oefp_oecluster_fingerprint_benchmark
   ./build-bench/benchmarks/oefp_oecluster_fingerprint_benchmark 512 0 256

Wheel Builds
------------

Build a wheel locally:

.. code-block:: bash

   python scripts/build_python.py --openeye-root /path/to/openeye/sdk --verbose

The wheel is written to ``dist/``. On macOS, the build script runs
``delocate``. On Linux, CI uses ``auditwheel``.

GitHub Actions
--------------

The release workflow builds wheels on:

- Linux x86_64
- Linux aarch64
- macOS
- Windows x64

It runs on version tags and ``workflow_dispatch``.

Required GitHub variables:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Variable
     - Description
   * - ``OPENEYE_VERSION``
     - OpenEye SDK version used by CI
   * - ``SDK_BUCKET``
     - Cloud storage bucket for SDK archives
   * - ``SDK_LINUX_X86_64``
     - Linux x86_64 SDK filename
   * - ``SDK_LINUX_AARCH64``
     - Linux aarch64 SDK filename
   * - ``SDK_MACOS``
     - macOS SDK filename
   * - ``SDK_WINDOWS``
     - Windows x64 SDK filename

Required GitHub secrets:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Secret
     - Description
   * - ``GCP_WORKLOAD_IDENTITY_PROVIDER``
     - GCP Workload Identity Federation provider
   * - ``GCP_SERVICE_ACCOUNT``
     - GCP service account for SDK downloads

Version Management
------------------

OEFP uses ``vrzn`` to keep version numbers synchronized:

.. code-block:: bash

   vrzn get
   vrzn bump patch
   vrzn set 1.0.0

The tracked version locations are configured in ``vrzn.toml``.
