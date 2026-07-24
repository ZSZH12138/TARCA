ifeq ($(OS),Windows_NT)
TARCA_CONDA_PREFIX ?= $(UV_PROJECT_ENVIRONMENT)
UV_CMD ?= $(TARCA_CONDA_PREFIX)/Scripts/uv.exe
export UV_PROJECT_ENVIRONMENT := $(TARCA_CONDA_PREFIX)
else
UV_CMD ?= uv
endif

.PHONY: lock sync doctor smoke test lint stage0-check

lock:
	"$(UV_CMD)" lock

sync:
	"$(UV_CMD)" sync --frozen --extra research --group dev

doctor:
	"$(UV_CMD)" run python scripts/doctor.py

smoke:
	"$(UV_CMD)" run pytest --no-cov -q tests/test_doctor.py tests/test_pot_smoke.py tests/test_torch_hook_smoke.py

test:
	"$(UV_CMD)" run pytest -q

lint:
	"$(UV_CMD)" run ruff check .
	"$(UV_CMD)" run ruff format --check .

stage0-check:
	"$(UV_CMD)" run python -m compileall -q src scripts tests third_party_manifest
	"$(UV_CMD)" run pytest -q
	"$(UV_CMD)" run pre-commit run --all-files
	"$(UV_CMD)" run python scripts/doctor.py
