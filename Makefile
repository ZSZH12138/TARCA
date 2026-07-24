ifeq ($(OS),Windows_NT)
TARCA_CONDA_PREFIX ?= $(UV_PROJECT_ENVIRONMENT)
UV ?= $(TARCA_CONDA_PREFIX)/Scripts/uv.exe
export UV_PROJECT_ENVIRONMENT := $(TARCA_CONDA_PREFIX)
else
UV ?= uv
endif

.PHONY: lock sync doctor smoke test lint stage0-check

lock:
	"$(UV)" lock

sync:
	"$(UV)" sync --frozen --extra research --group dev

doctor:
	"$(UV)" run python scripts/doctor.py

smoke:
	"$(UV)" run pytest --no-cov -q tests/test_doctor.py tests/test_pot_smoke.py tests/test_torch_hook_smoke.py

test:
	"$(UV)" run pytest -q

lint:
	"$(UV)" run ruff check .
	"$(UV)" run ruff format --check .

stage0-check:
	"$(UV)" run python -m compileall -q src scripts tests third_party_manifest
	"$(UV)" run pytest -q
	"$(UV)" run pre-commit run --all-files
	"$(UV)" run python scripts/doctor.py
