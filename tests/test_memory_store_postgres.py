"""Integration test: the Postgres (pgvector) memory store.

Runs only when TEST_DATABASE_URL points at a real pgvector Postgres. Verifies
get_memory_store() returns an AsyncPostgresStore (no startup load / re-embed)
and that aput + asearch round-trip. A deterministic fake embedder is used so the
test exercises the store wiring without calling Vertex.
"""
import os
from unittest.mock import patch

import pytest

PG_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not PG_URL, reason="set TEST_DATABASE_URL to a pgvector Postgres URL to run"
)


def _fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * 768
        v[hash(t) % 768] = 1.0
        out.append(v)
    return out


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    import memory

    monkeypatch.setenv("DATABASE_URL", PG_URL)
    for a in ("_store", "_store_pool"):
        setattr(memory, a, None)
    yield
    # Throwaway container handles connection cleanup; just drop the singletons.
    for a in ("_store", "_store_pool"):
        setattr(memory, a, None)


async def test_get_memory_store_uses_postgres_and_round_trips():
    import memory

    with patch("memory._create_embed_fn", return_value=_fake_embed):
        store = await memory.get_memory_store()

    assert type(store).__name__ == "AsyncPostgresStore"

    ns = ("core",)
    await store.aput(ns, "k-deploy", {"content": "deploy via blue-green health gate"})
    await store.aput(ns, "k-logs", {"content": "check cloud run revision logs after deploy"})

    results = await store.asearch(ns, query="deploy", limit=2)
    assert len(results) >= 1
    assert any("blue-green" in r.value["content"] for r in results)
