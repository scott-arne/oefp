Installation
============

OEFP can be installed as a Python package or built from source as a C++ library
with Python bindings.

Prerequisites
-------------

- **C++ Compiler**: GCC 9+, Clang 10+, or MSVC 2019+
- **CMake**: 3.16 or later
- **OpenEye Toolkit**: Licensed OpenEye C++ SDK for building, Python package for runtime
- **Python**: 3.10 or later
- **SWIG**: 4.0 or later
- **RDKit**: Required for conformance tests, not required by the runtime package

Python Package
--------------

Install OpenEye Toolkits first:

.. code-block:: bash

   pip install --extra-index-url https://pypi.anaconda.org/openeye/simple openeye-toolkits

Install OEFP:

.. code-block:: bash

   pip install oefp

The ``oefp`` package includes compiled C++ extensions. Wheels are built for the
OpenEye runtime expected by the package release.

C++ Library
-----------

Build from source using CMake:

.. code-block:: bash

   git clone https://github.com/scott-arne/oefp.git
   cd oefp

   export OPENEYE_ROOT=/path/to/openeye/sdk
   cmake --preset release
   cmake --build build-release --parallel

Run the C++ tests from a debug build:

.. code-block:: bash

   cmake --preset debug
   cmake --build build-debug --target oefp_tests
   ctest --test-dir build-debug --output-on-failure

Development Install
-------------------

Install the Python package in editable mode:

.. code-block:: bash

   pip install --config-settings editable_mode=compat -e python/

The ``editable_mode=compat`` flag keeps the package on a traditional editable
path that works with the compiled SWIG extension. Rebuild the native target
after C++ changes:

.. code-block:: bash

   cmake --build build-debug --target oefp_python

CMake Options
-------------

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``OEFP_BUILD_TESTS``
     - ON
     - Build C++ unit tests
   * - ``OEFP_BUILD_PYTHON``
     - ON
     - Build Python SWIG bindings
   * - ``OEFP_BUILD_BENCHMARKS``
     - OFF
     - Build optional benchmark executables
   * - ``OEFP_OECLUSTER_SOURCE_DIR``
     - empty
     - Local ``oecluster`` checkout used by optional benchmarks
   * - ``OEFP_USE_STABLE_ABI``
     - ON
     - Build Python bindings with the stable ABI
   * - ``OEFP_UNIVERSAL2``
     - OFF
     - Build macOS universal2 binaries
   * - ``OPENEYE_ROOT``
     - required for source builds
     - Path to the OpenEye C++ SDK

Troubleshooting
---------------

OpenEye License
^^^^^^^^^^^^^^^

Ensure your OpenEye license is configured:

.. code-block:: bash

   export OE_LICENSE=/path/to/oe_license.txt

Import Errors
^^^^^^^^^^^^^

If Python cannot import ``oefp``, verify:

1. OpenEye Toolkits are installed in the active Python environment.
2. The OEFP wheel matches the installed OpenEye runtime.
3. The native extension has been rebuilt after local C++ changes.

Version Mismatch
^^^^^^^^^^^^^^^^

If OEFP reports an OpenEye runtime mismatch, reinstall the package for the
active OpenEye environment:

.. code-block:: bash

   pip install --force-reinstall oefp
