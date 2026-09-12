from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from loom_memory.service import MemoryService, build_service
from loom_memory.settings import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", allowed_roots=[tmp_path])


@pytest.fixture
async def service(settings: Settings) -> AsyncIterator[MemoryService]:
    svc = build_service(settings, fake=True)
    await svc.start()
    yield svc
    await svc.close()
