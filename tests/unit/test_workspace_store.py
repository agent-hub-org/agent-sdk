import pytest
from agent_sdk.workspace.store import WorkspaceStore


@pytest.mark.asyncio
async def test_write_and_read_in_memory():
    store = WorkspaceStore(redis_url=None)
    await store.init()
    await store.write("ws1", "macro", {"findings": "bullish", "confidence": 0.9})
    result = await store.read("ws1", "macro")
    assert result == {"findings": "bullish", "confidence": 0.9}


@pytest.mark.asyncio
async def test_read_missing_key_returns_none():
    store = WorkspaceStore(redis_url=None)
    await store.init()
    assert await store.read("nonexistent", "macro") is None


@pytest.mark.asyncio
async def test_flush_removes_all_workspace_keys():
    store = WorkspaceStore(redis_url=None)
    await store.init()
    await store.write("ws2", "macro", {"data": 1})
    await store.write("ws2", "sector", {"data": 2})
    await store.write("other_ws", "macro", {"data": 3})
    await store.flush("ws2")
    assert await store.read("ws2", "macro") is None
    assert await store.read("ws2", "sector") is None
    # other workspace untouched
    assert await store.read("other_ws", "macro") == {"data": 3}


@pytest.mark.asyncio
async def test_degraded_flag_true_when_no_redis_url():
    store = WorkspaceStore(redis_url=None)
    await store.init()
    assert store._degraded is True


@pytest.mark.asyncio
async def test_write_overwrites_existing_key():
    store = WorkspaceStore(redis_url=None)
    await store.init()
    await store.write("ws3", "macro", {"findings": "old"})
    await store.write("ws3", "macro", {"findings": "new"})
    result = await store.read("ws3", "macro")
    assert result["findings"] == "new"
