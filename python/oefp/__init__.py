"""
OEFP - Improved fingerprints for the OpenEye Toolkits
"""

import os
import re
import warnings

__version__ = "0.1.0"
__version_info__ = (0, 1, 0)


def _find_openeye_runtime_lib_dir(expected_libs=()):
    """Find the OpenEye runtime library directory without importing oechem."""
    import importlib.util
    import sys
    from pathlib import Path

    search_locations = []
    openeye_module = sys.modules.get("openeye")
    openeye_path = getattr(openeye_module, "__path__", None)
    if openeye_path is not None:
        search_locations.extend(openeye_path)

    if not search_locations:
        try:
            openeye_spec = importlib.util.find_spec("openeye")
        except (ImportError, ValueError):
            openeye_spec = None
        if (
            openeye_spec is not None
            and openeye_spec.submodule_search_locations is not None
        ):
            search_locations.extend(openeye_spec.submodule_search_locations)

    expected_libs = set(expected_libs or ())
    fallback_dir = None
    for package_root in search_locations:
        libs_root = Path(package_root) / "libs"
        if not libs_root.is_dir():
            continue

        # Importing openeye.libs eagerly imports oechem in some environments.
        # The runtime libraries are shipped below openeye/libs, so filesystem
        # discovery preserves the fresh-import condition.
        for root, _, files in os.walk(libs_root):
            file_set = set(files)
            if expected_libs and expected_libs.intersection(file_set):
                return root
            if fallback_dir is None and any(
                ".dylib" in lib_name or ".so" in lib_name or ".dll" in lib_name
                for lib_name in files
            ):
                fallback_dir = root

    return fallback_dir


def _ensure_library_compat():
    """Create compatibility symlinks when OpenEye library versions differ from build time.

    When this package is built with shared OpenEye libraries, the compiled extension
    records the exact versioned library filenames (e.g., liboechem-4.3.0.1.dylib).
    If the user upgrades openeye-toolkits, these filenames change and the dynamic
    linker fails to load the extension.

    This function detects version mismatches and creates symlinks from the expected
    (build-time) library names to the actual (runtime) library files.
    """
    try:
        from . import _build_info
    except ImportError:
        return False

    if getattr(_build_info, 'OPENEYE_LIBRARY_TYPE', 'STATIC') != 'SHARED':
        return False

    expected_libs = getattr(_build_info, 'OPENEYE_EXPECTED_LIBS', [])
    if not expected_libs:
        return False

    oe_lib_dir = _find_openeye_runtime_lib_dir(expected_libs)
    if oe_lib_dir is None:
        return False

    if not os.path.isdir(oe_lib_dir):
        return False

    pkg_dir = os.path.dirname(__file__)
    created_any = False

    for expected_name in expected_libs:
        if os.path.exists(os.path.join(oe_lib_dir, expected_name)):
            continue

        symlink_path = os.path.join(pkg_dir, expected_name)
        if os.path.islink(symlink_path):
            if os.path.exists(symlink_path):
                continue
            try:
                os.unlink(symlink_path)
            except OSError:
                continue
        elif os.path.exists(symlink_path):
            continue

        match = re.match(r'(lib\w+?)(-[\d.]+)?(\.[\d.]*\w+)$', expected_name)
        if not match:
            continue
        base_name = match.group(1)

        actual_path = None
        for f in os.listdir(oe_lib_dir):
            if f.startswith(base_name + '-') or f.startswith(base_name + '.'):
                actual_path = os.path.join(oe_lib_dir, f)
                break

        if actual_path:
            try:
                os.symlink(actual_path, os.path.join(pkg_dir, expected_name))
                created_any = True
            except OSError:
                pass

    return created_any


def _preload_shared_libs():
    """Preload OpenEye shared libraries so the C extension can find them.

    On Linux, the extension's RUNPATH (set at build time) normally handles
    dependency resolution, but preloading ensures libraries are available
    even if RUNPATH is stripped (e.g. by certain packaging tools).
    On macOS, @rpath references may not resolve without preloading.

    Only the libraries recorded in ``OPENEYE_EXPECTED_LIBS`` are loaded,
    and they are loaded with ``RTLD_GLOBAL`` so that cross-module C++
    symbol references resolve correctly. Loading the entire OpenEye
    library directory (which can contain 70+ unrelated shared objects)
    would pollute the global symbol namespace and cause segfaults in
    unrelated C extensions such as ``_sqlite3``.
    """
    import ctypes
    import sys
    if sys.platform not in ('linux', 'darwin'):
        return

    try:
        from . import _build_info
    except ImportError:
        return

    if getattr(_build_info, 'OPENEYE_LIBRARY_TYPE', 'STATIC') != 'SHARED':
        return

    expected_libs = getattr(_build_info, 'OPENEYE_EXPECTED_LIBS', [])
    if not expected_libs:
        return

    oe_lib_dir = _find_openeye_runtime_lib_dir(expected_libs)
    if oe_lib_dir is None:
        return

    if not os.path.isdir(oe_lib_dir):
        return

    pkg_dir = os.path.dirname(__file__)
    for lib_name in expected_libs:
        # Try the OpenEye lib directory first, then local symlinks
        oe_path = os.path.join(oe_lib_dir, lib_name)
        local_path = os.path.join(pkg_dir, lib_name)
        path = oe_path if os.path.exists(oe_path) else local_path
        if os.path.exists(path) or os.path.islink(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def _preload_bundled_libs():
    """Preload libraries bundled by auditwheel from the .libs directory.

    auditwheel repair bundles non-manylinux dependencies (e.g. libraries
    from FetchContent or system packages) into a ``<package>.libs/``
    directory next to the package. The bundled copies have hashed filenames
    and must be loaded before the C extension to satisfy its DT_NEEDED
    entries.

    Libraries may have inter-dependencies, so we do multiple passes
    until no new libraries can be loaded. Libraries are loaded without
    ``RTLD_GLOBAL`` to avoid polluting the global symbol namespace.
    """
    import sys
    if sys.platform != 'linux':
        return

    import ctypes
    pkg_name = __name__
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    site_dir = os.path.dirname(pkg_dir)
    for libs_name in (f'{pkg_name}.libs', f'.{pkg_name}.libs'):
        libs_dir = os.path.join(site_dir, libs_name)
        if not os.path.isdir(libs_dir):
            continue
        remaining = [
            os.path.join(libs_dir, f)
            for f in sorted(os.listdir(libs_dir))
            if '.so' in f
        ]
        while remaining:
            failed = []
            for lib_path in remaining:
                try:
                    ctypes.CDLL(lib_path)
                except OSError:
                    failed.append(lib_path)
            if len(failed) == len(remaining):
                break
            remaining = failed


def _check_openeye_version():
    """Check that the OpenEye version matches what was used at build time."""
    try:
        from . import _build_info
    except ImportError:
        return

    if getattr(_build_info, 'OPENEYE_LIBRARY_TYPE', 'STATIC') != 'SHARED':
        return

    build_version = getattr(_build_info, 'OPENEYE_BUILD_VERSION', None)
    if not build_version:
        return

    try:
        from importlib import metadata
        runtime_version = metadata.version("openeye-toolkits")
    except metadata.PackageNotFoundError:
        warnings.warn(
            "openeye-toolkits package not found. "
            "Install with: pip install openeye-toolkits",
            ImportWarning
        )
        return

    build_parts = build_version.split('.')[:2]
    runtime_parts = runtime_version.split('.')[:2]
    if build_parts != runtime_parts:
        warnings.warn(
            f"OpenEye version mismatch: oefp was built with "
            f"OpenEye Toolkits {build_version} but runtime has {runtime_version}. "
            f"This may cause compatibility issues.",
            RuntimeWarning
        )


_ensure_library_compat()
_preload_shared_libs()
_preload_bundled_libs()
_check_openeye_version()

from .oefp import (
    calculate_molecular_weight,
)
from .api import (
    Metric,
    OEFP,
    OEFPBatch,
    OEFPCount,
    OEFPCountBatch,
    OEFPMappingSet,
    OEFPSparse,
    OEFPSparseBatch,
    MorganCountFingerprintResult,
    MorganFingerprintResult,
    MorganSparseCountFingerprintResult,
    MorganSparseFingerprintResult,
    atom_pair_fingerprint,
    cdist,
    compare,
    from_openeye_fingerprint,
    morgan_count_fingerprint,
    morgan_count_fingerprint_with_mapping,
    morgan_fingerprint,
    morgan_fingerprint_with_mapping,
    morgan_sparse_count_fingerprint,
    morgan_sparse_count_fingerprint_with_mapping,
    morgan_sparse_fingerprint,
    morgan_sparse_fingerprint_with_mapping,
    pdist,
    to_openeye_fingerprint,
)

__all__ = [
    "__version__",
    "__version_info__",
    "Metric",
    "OEFP",
    "OEFPBatch",
    "OEFPCount",
    "OEFPCountBatch",
    "OEFPMappingSet",
    "OEFPSparse",
    "OEFPSparseBatch",
    "MorganCountFingerprintResult",
    "MorganFingerprintResult",
    "MorganSparseCountFingerprintResult",
    "MorganSparseFingerprintResult",
    "atom_pair_fingerprint",
    "calculate_molecular_weight",
    "cdist",
    "compare",
    "from_openeye_fingerprint",
    "morgan_count_fingerprint",
    "morgan_count_fingerprint_with_mapping",
    "morgan_fingerprint",
    "morgan_fingerprint_with_mapping",
    "morgan_sparse_count_fingerprint",
    "morgan_sparse_count_fingerprint_with_mapping",
    "morgan_sparse_fingerprint",
    "morgan_sparse_fingerprint_with_mapping",
    "pdist",
    "to_openeye_fingerprint",
]
