"""Package import behavior tests."""

import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_pyarrow_is_required_for_descriptor_storage():
    """Descriptor storage relies on the PyArrow-distributed Arrow runtime."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    assert pa.__version__
    assert pq is not None


def test_oefp_import_preloads_pyarrow_runtime():
    """A cold OEFP import should not require users to import PyArrow first."""
    if not list(Path("python/oefp").glob("_oefp*")):
        pytest.skip("native extension has not been built")

    env = os.environ.copy()
    python_path = str(Path("python").resolve())
    if env.get("PYTHONPATH"):
        python_path = f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = python_path

    result = subprocess.run(
        [sys.executable, "-c", "import oefp; print(oefp.__version__)"],
        check=False,
        cwd=Path.cwd(),
        env=env,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_import_uses_user_cache_for_broken_openeye_runtime_compat_symlink(
    monkeypatch,
    tmp_path,
):
    """Broken OpenEye runtime symlinks should not mutate oefp."""
    package = "oefp"
    source_dir = tmp_path / package
    shutil.copytree(
        "python/oefp",
        source_dir,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "_*.so",
            "_*.pyd",
            "_*.dylib",
            "lib*.so",
            "lib*.dylib",
            "lib*.a",
        ),
    )
    expected_name = "liboechem-4.3.0.1.so"
    runtime_name = "liboechem-4.3.0.3.so"

    (source_dir / "_build_info.py").write_text(
        "OPENEYE_LIBRARY_TYPE = 'SHARED'\n"
        f"OPENEYE_EXPECTED_LIBS = [{expected_name!r}]\n"
        "OPENEYE_BUILD_VERSION = '2025.2.1'\n"
    )
    (source_dir / "api.py").write_text(
        'class _Stub:\n    def __init__(self, *args, **kwargs):\n        pass\n    def __call__(self, *args, **kwargs):\n        return None\n\ndef __getattr__(name):\n    return _Stub\n'
    )

    fake_openeye = tmp_path / "openeye"
    fake_libs = fake_openeye / "libs"
    fake_runtime = fake_libs / "python3-linux-x64-g++10.x"
    fake_runtime.mkdir(parents=True)
    (fake_openeye / "__init__.py").write_text("")
    marker = tmp_path / "openeye_imported.txt"
    (fake_libs / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('libs')\n"
    )
    (fake_openeye / "oechem.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('oechem')\n"
    )
    (fake_runtime / runtime_name).write_text("not a real library")
    (fake_runtime / expected_name).symlink_to(fake_runtime / "missing-liboechem.so")
    cache_home = tmp_path / "cache"

    for module_name in list(sys.modules):
        if module_name == package or module_name.startswith(f"{package}."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
        if module_name == "openeye" or module_name.startswith("openeye."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    monkeypatch.setattr(
        sys,
        "meta_path",
        [
            finder
            for finder in sys.meta_path
            if package not in type(finder).__module__
        ],
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    importlib.invalidate_caches()

    importlib.import_module(package)

    assert not marker.exists()
    assert "openeye.libs" not in sys.modules
    assert "openeye.oechem" not in sys.modules
    assert not (source_dir / expected_name).exists()
    cached_aliases = list(
        cache_home.glob(f"{package}/openeye-libs/**/{expected_name}")
    )
    assert len(cached_aliases) == 1
    assert cached_aliases[0].is_symlink()
    assert cached_aliases[0].resolve().name == runtime_name


def test_import_mirrors_transitive_openeye_runtime_into_cache(
    monkeypatch,
    tmp_path,
):
    """Relocating the extension must keep transitive OpenEye libs reachable.

    When a name-drift alias forces the extension to load from the user cache,
    its ``$ORIGIN`` no longer reaches the OpenEye library directory. Transitive
    dependencies that are not recorded in ``OPENEYE_EXPECTED_LIBS`` (such as
    ``liboecuda``) must therefore be mirrored into the cache, or the dynamic
    linker fails to load the extension.
    """
    package = "oefp"
    source_dir = tmp_path / package
    shutil.copytree(
        "python/oefp",
        source_dir,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "_*.so",
            "_*.pyd",
            "_*.dylib",
            "lib*.so",
            "lib*.dylib",
            "lib*.a",
        ),
    )
    expected_name = "liboechem-4.3.0.1.so"
    runtime_name = "liboechem-4.3.0.3.so"
    # A transitive dependency the build never records. It exists in the OpenEye
    # runtime directory but is not a direct dependency of the extension.
    transitive_name = "liboecuda-2.3.1.3.so"

    (source_dir / "_build_info.py").write_text(
        "OPENEYE_LIBRARY_TYPE = 'SHARED'\n"
        f"OPENEYE_EXPECTED_LIBS = [{expected_name!r}]\n"
        "OPENEYE_BUILD_VERSION = '2025.2.1'\n"
    )
    (source_dir / "api.py").write_text(
        'class _Stub:\n    def __init__(self, *args, **kwargs):\n        pass\n    def __call__(self, *args, **kwargs):\n        return None\n\ndef __getattr__(name):\n    return _Stub\n'
    )

    fake_openeye = tmp_path / "openeye"
    fake_libs = fake_openeye / "libs"
    fake_runtime = fake_libs / "python3-linux-x64-g++10.x"
    fake_runtime.mkdir(parents=True)
    (fake_openeye / "__init__.py").write_text("")
    (fake_libs / "__init__.py").write_text(
        "raise AssertionError('openeye.libs should not be imported')\n"
    )
    (fake_runtime / runtime_name).write_text("not a real library")
    (fake_runtime / transitive_name).write_text("not a real library")
    # The expected name drifted: only a broken alias at the recorded filename.
    (fake_runtime / expected_name).symlink_to(fake_runtime / "missing-liboechem.so")
    cache_home = tmp_path / "cache"

    for module_name in list(sys.modules):
        if module_name == package or module_name.startswith(f"{package}."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
        if module_name == "openeye" or module_name.startswith("openeye."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    monkeypatch.setattr(
        sys,
        "meta_path",
        [
            finder
            for finder in sys.meta_path
            if package not in type(finder).__module__
        ],
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    importlib.invalidate_caches()

    importlib.import_module(package)

    # The transitive library is symlinked into the same cache directory as the
    # drift alias, restoring a complete $ORIGIN-local view of the runtime set.
    drift_aliases = list(
        cache_home.glob(f"{package}/openeye-libs/**/{expected_name}")
    )
    assert len(drift_aliases) == 1
    cache_dir = drift_aliases[0].parent

    transitive_alias = cache_dir / transitive_name
    assert transitive_alias.is_symlink()
    assert transitive_alias.resolve() == (fake_runtime / transitive_name).resolve()

    # The drifted direct dependency is mirrored under its real name as well.
    runtime_alias = cache_dir / runtime_name
    assert runtime_alias.is_symlink()
    assert runtime_alias.resolve() == (fake_runtime / runtime_name).resolve()
