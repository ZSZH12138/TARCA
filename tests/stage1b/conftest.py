from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "official_source: requires hash-pinned official Stage1B source checkouts",
    )
