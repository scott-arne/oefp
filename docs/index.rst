OEFP Documentation
==================

**RDKit-compatible fingerprints and descriptor tables for OpenEye molecules**
- A high-performance C++ library with Python bindings for generating, storing,
and comparing molecular fingerprints from OpenEye ``OEMolBase`` objects.
OEFP also exposes raw generator descriptors and schema-backed
Mordred-compatible descriptor rows.

.. note::

   This documentation covers the C++ core library and Python bindings.

Overview
--------

OEFP provides tools for:

- **Fingerprint Generation**: RDKit-compatible Morgan and topological Atom Pair
  fingerprints from OpenEye molecules
- **Fingerprint Storage**: Dense binary, sparse binary, and sparse counted
  containers
- **Batch Comparison**: Scalar, query-to-batch, ``cdist``, and condensed
  ``pdist`` kernels
- **Descriptor Tables**: Raw counted Morgan and topological Atom Pair
  descriptors, plus schema-backed Mordred-compatible descriptor rows with
  missing-value support and Arrow/Parquet interchange
- **Prerequisite Metadata**: Descriptor definitions can declare lightweight
  prerequisite bitmaps such as existing 2D or 3D coordinates
- **OpenEye Integration**: Native OpenEye molecule and ``OEFingerPrint``
  interoperability

Key Features
------------

- **RDKit-Compatible Output**: Morgan and topological Atom Pair generators are tested
  against RDKit for supported options
- **Fast Batch Kernels**: Contiguous batch storage with cached popcounts for
  high-throughput similarity and distance calculations
- **Python Bindings**: SWIG-based Python bindings with read-only NumPy views
  over C++-owned memory
- **Explicit Scientific Boundaries**: Descriptor calculators do not generate
  2D or 3D coordinates implicitly; descriptors whose prerequisites are absent
  remain unavailable

Quick Links
-----------

- :doc:`getting-started/installation` - Install OEFP
- :doc:`getting-started/quickstart` - Generate and compare fingerprints
- :ref:`api-comparison` - Compare RDKit, OEFP, and OEGraphSim APIs
- :doc:`cpp-api/library_root` - C++ API Reference
- :doc:`python-lib/index` - Python Library Reference

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started/installation
   getting-started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: C++ Library

   cpp-api/library_root

.. toctree::
   :maxdepth: 2
   :caption: Python Library

   python-lib/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer-guide/cicd
   developer-guide/descriptor-deduplication

Indices and Tables
==================

* :ref:`genindex`
* :ref:`search`
