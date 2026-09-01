default: typecheck

# Check the distributable package with every supported type checker.
typecheck:
    #!/usr/bin/env bash
    set -uo pipefail
    status=0
    just basedpyright || status=1
    just basedmypy || status=1
    just pyrefly || status=1
    just ty || status=1
    exit "$status"

basedpyright:
    uv run basedpyright src

basedmypy:
    uv run mypy src

pyrefly:
    uv run pyrefly check src

ty:
    uv run ty check src
