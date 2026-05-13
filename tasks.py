"""Invoke tasks for OEFP project management."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from invoke import task

PROJECT_ROOT = Path(__file__).parent.absolute()
DOCS_DIR = PROJECT_ROOT / "docs"
BUILD_DIR = DOCS_DIR / "_build"
HTML_DIR = BUILD_DIR / "html"


@task
def docs(ctx, clean=False):
    """Build Sphinx documentation.

    :param clean: Remove build directory first.
    """
    os.chdir(DOCS_DIR)

    if clean:
        print("Cleaning build directory...")
        ctx.run("make clean", warn=True)

    print("Building documentation...")
    result = ctx.run("make html", warn=True)

    if result.ok:
        print("\nDocumentation built successfully.")
        print(f"Open: file://{HTML_DIR}/index.html")
    else:
        print("\nDocumentation build failed.", file=sys.stderr)
        sys.exit(1)


@task
def serve_docs(ctx, port=8000, clean=False, watch=False):
    """Build documentation and serve at localhost.

    :param port: Port to serve on.
    :param clean: Remove build directory first.
    :param watch: Rebuild automatically when docs files change.
    """
    if clean:
        os.chdir(DOCS_DIR)
        print("Cleaning build directory...")
        ctx.run("make clean", warn=True)

    if watch:
        print(f"\nWatching for changes and serving at http://localhost:{port}")
        print("Press Ctrl+C to stop.\n")
        os.chdir(DOCS_DIR)
        ctx.run(
            f"{sys.executable} -m sphinx_autobuild"
            f" . {HTML_DIR}"
            f" --port {port}"
            f" --open-browser"
            f" --re-ignore '/_doxygen/'"
            f" --re-ignore '/cpp-api/'"
            f" --re-ignore '/_build/'"
        )
    else:
        docs(ctx, clean=False)

        print(f"\nServing documentation at http://localhost:{port}")
        print("Press Ctrl+C to stop.\n")

        os.chdir(HTML_DIR)
        ctx.run(f"{sys.executable} -m http.server {port}")


@task
def docs_check(ctx):
    """Build documentation with warnings as errors."""
    os.chdir(DOCS_DIR)

    print("Building documentation with strict checking...")
    result = ctx.run("make check", warn=True)

    if result.ok:
        print("\nDocumentation check passed.")
    else:
        print("\nDocumentation check failed.", file=sys.stderr)
        sys.exit(1)


@task
def docs_deps(ctx):
    """Install documentation dependencies."""
    print("Installing documentation dependencies...")
    ctx.run(f"{sys.executable} -m pip install -r {DOCS_DIR}/requirements.txt")
    print("Done.")
