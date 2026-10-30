"""Filter trees and their JSON: what other engines are handed.

What has to hold: every node and region survives a round trip through JSON
text unchanged, a leaf's cache key changes exactly when its content does,
and a malformed region or document is refused when it is built or read
rather than giving a wrong answer later.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from zarr_vectors_tools.filtering import (
    All,
    And,
    Box,
    Column,
    Disk,
    InGroup,
    Mask,
    Not,
    Or,
    Passes,
    Slab,
    Sphere,
    from_json,
    to_json,
)


def _mask() -> Mask:
    voxels = np.zeros((3, 4, 5), dtype=bool)
    voxels[1, 2, 3] = voxels[0, 0, 0] = voxels[2, 3, 4] = True
    affine = np.array([[0, 0.5, 0, 1], [-0.5, 0, 0, 2], [0, 0, 2, 3], [0, 0, 0, 1]], float)
    return Mask(voxels, affine, spacing=0.1)


def _every_kind():
    return Or(
        And(
            Passes(Sphere((1, 2, 3), 4.5)),
            Passes(Box((0, 0, 0), (1, 2, 3)), "endpoints"),
            Not(Passes(Slab(2, -1.5, 1.5), "both_endpoints")),
        ),
        Passes(Disk((0, 0, 0), (0, 0, 3), 2.0, (10, 80))),
        Passes(_mask()),
        Column("length", "between", (40, 200)) - Column("fa", "absent"),
        Column("label", "in", [5, 3, 5]),
        Column("rgb", ">=", 0.25, channel=1),
        InGroup((4, 1, 4)),
        All(),
    )


def test_every_node_and_region_round_trips_through_json_text() -> None:
    tree = _every_kind()
    text = json.dumps(to_json(tree))
    back = from_json(text)
    assert back == tree
    assert to_json(back) == to_json(tree)
    assert back.key() == tree.key()


def test_the_document_is_versioned() -> None:
    doc = to_json(All())
    assert doc == {"format": "zvf-predicate", "version": 1, "tree": {"op": "all"}}
    with pytest.raises(ValueError, match="version 2 is not supported"):
        from_json({**doc, "version": 2})
    with pytest.raises(ValueError, match="not a zvf-predicate"):
        from_json({"tree": {"op": "all"}})
    with pytest.raises(ValueError, match="unknown node op"):
        from_json({**doc, "tree": {"op": "xor"}})
    with pytest.raises(ValueError, match="unknown region"):
        from_json({**doc, "tree": {"op": "passes", "region": {"type": "cone"}}})


def test_values_are_normalised_on_construction() -> None:
    disk = Disk((0, 0, 0), (0, 0, 7), 1)
    assert disk.normal == (0.0, 0.0, 1.0)
    assert Column("label", "in", [3, 1, 3]).value == (1, 3)
    assert InGroup([2, 0, 2]).ids == (0, 2)
    assert Column("fa", "absent", 12).value is None


def test_keys_follow_content() -> None:
    a = Passes(Sphere((0, 0, 0), 5))
    assert a.key() == Passes(Sphere((0.0, 0.0, 0.0), 5.0)).key()
    assert a.key() != Passes(Sphere((0, 0, 0), 5.0001)).key()
    assert a.key() != Passes(Sphere((0, 0, 0), 5), "endpoints").key()
    moved = _mask()
    voxels = moved.voxels.copy()
    voxels[1, 1, 1] = True
    assert Passes(Mask(voxels, moved.affine, 0.1)).key() != Passes(moved).key()


def test_operators_build_trees() -> None:
    a, b = Passes(Sphere((0, 0, 0), 1)), Column("length", ">", 1)
    assert a & b == And(a, b)
    assert a | b == Or(a, b)
    assert ~a == Not(a)
    assert a - b == And(a, Not(b))
    assert (a - b).leaves() == [a, b]


@pytest.mark.parametrize("build, message", [
    (lambda: Sphere((0, 0, 0), -1), "radius must be 0 or more"),
    (lambda: Sphere((0, 0), 1), "three finite numbers"),
    (lambda: Sphere((0, 0, np.nan), 1), "three finite numbers"),
    (lambda: Box((1, 1, 1), (0, 2, 2)), "below its min corner"),
    (lambda: Slab(3, 0, 1), "axis must be 0, 1 or 2"),
    (lambda: Slab(0, 1, 0), "below its min"),
    (lambda: Disk((0, 0, 0), (0, 0, 0), 1), "zero vector"),
    (lambda: Disk((0, 0, 0), (0, 0, 1), 1, (50, 40)), "lo <= hi <= 90"),
    (lambda: Mask(np.ones((2, 2)), np.eye(4)), "must be 3-D"),
    (lambda: Mask(np.ones((2, 2, 2)), np.eye(4), spacing=0), "spacing must be above 0"),
    (lambda: Passes(Sphere((0, 0, 0), 1), "middle"), "mode must be"),
    (lambda: Passes(Disk((0, 0, 0), (0, 0, 1), 1), "endpoints"), "only mode='path'"),
    (lambda: Column("fa", "~", 1), "cmp must be"),
    (lambda: Column("fa", "between", (2, 1)), "lo <= hi"),
    (lambda: InGroup((-1,)), "0 or more"),
    (lambda: And(), "at least one operand"),
])
def test_refusals(build, message) -> None:
    with pytest.raises(ValueError, match=message):
        build()
