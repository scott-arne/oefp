OEFP Documentation
==================

**RDKit-compatible fingerprints for OpenEye molecules** - A high-performance
C++ library with Python bindings for generating, storing, and comparing
molecular fingerprints from OpenEye ``OEMolBase`` objects.

.. note::

   This documentation covers the C++ core library and Python bindings.

Overview
--------

OEFP provides tools for:

- **Fingerprint Generation**: RDKit-compatible Morgan and Atom Pair
  fingerprints from OpenEye molecules
- **Fingerprint Storage**: Dense binary, sparse binary, and sparse counted
  containers
- **Batch Comparison**: Scalar, query-to-batch, ``cdist``, and condensed
  ``pdist`` kernels
- **OpenEye Integration**: Native OpenEye molecule and ``OEFingerPrint``
  interoperability

Key Features
------------

- **RDKit-Compatible Output**: Morgan and Atom Pair generators are tested
  against RDKit for supported options
- **Fast Batch Kernels**: Contiguous batch storage with cached popcounts for
  high-throughput similarity and distance calculations
- **Python Bindings**: SWIG-based Python bindings with read-only NumPy views
  over C++-owned memory

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

Indices and Tables
==================

* :ref:`genindex`
* :ref:`search`
