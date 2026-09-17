"""The stdlib process-pool executor: one pool per run, one payload per phase.

The parity tests elsewhere (``test_polyline_coarsen_chunk_local``,
``test_skeleton_coarsen``) drive the coarseners through a pool of their own,
so nothing exercised :func:`process_pool_executor` itself -- the thing
``zvtools convert --workers N`` actually runs on.  These do.
"""

from __future__ import annotations

import os

import pytest

from zarr_vectors_tools.convert.ingest._parallel import (
    _call_with_shared,
    _map_chunksize,
    _set_shared,
    process_pool_executor,
)


def _scaled(item, shared=None):
    """Module level so a spawned worker can import it."""
    return item * (shared or {}).get("scale", 1)


def _worker_pid(item, shared=None):
    return os.getpid()


def _payload_identity(item, shared=None):
    return None if shared is None else id(shared)


def test_shared_payload_reaches_the_worker() -> None:
    _set_shared({"scale": 3})
    try:
        assert _call_with_shared(_scaled, 5) == 15
    finally:
        _set_shared(None)


def test_chunksize_batches_only_when_there_is_something_to_batch() -> None:
    assert _map_chunksize(10, 4) == 1
    assert _map_chunksize(5_000, 12) == 32
    assert _map_chunksize(200, 4) == 12


@pytest.mark.slow
def test_phases_with_different_payloads_share_one_pool() -> None:
    """Three phases, three payloads, results in input order -- and the same
    worker processes throughout.  The old executor opened a pool per payload
    and kept every one of them until the context closed."""
    with process_pool_executor(2) as ex:
        assert ex(_scaled, [1, 2, 3], {"scale": 10}) == [10, 20, 30]
        assert ex(_scaled, [1, 2, 3], {"scale": 100}) == [100, 200, 300]
        # A phase with no payload must not inherit the previous one.
        assert ex(_scaled, [1, 2, 3], None) == [1, 2, 3]
        pids_a = set(ex(_worker_pid, list(range(40)), {"phase": "a"}))
        pids_b = set(ex(_worker_pid, list(range(40)), {"phase": "b"}))
    assert pids_a and pids_a == pids_b
    assert len(pids_a) <= 2


@pytest.mark.slow
def test_payload_is_loaded_once_per_worker_per_phase() -> None:
    """Within one phase every task on a worker sees the same payload object:
    the file is read on the first task carrying the phase's token and kept,
    not re-read per task."""
    with process_pool_executor(2) as ex:
        ids_a = ex(_payload_identity, list(range(30)), {"k": 1})
        none_ids = ex(_payload_identity, list(range(5)), None)
    # One payload object per worker in a phase -> at most two distinct ids.
    assert 1 <= len(set(ids_a)) <= 2
    assert none_ids == [None] * 5


@pytest.mark.slow
def test_empty_item_list_is_a_no_op() -> None:
    with process_pool_executor(2) as ex:
        assert ex(_scaled, [], {"scale": 2}) == []
