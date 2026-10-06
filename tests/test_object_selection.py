"""Tests for `apply_sparsity`'s `alive_mask` parameter.

Without `alive_mask`, `"random"` sparsity samples uniformly over the full
*original* object-index space at every pyramid level.  Once earlier levels
have emptied (but not removed — object IDs are stable across levels) most
of that space, a naive random draw wastes most of its budget re-"keeping"
already-empty objects, silently collapsing survivor counts far below the
requested fraction.  `"length"` (and friends) dodge this by accident, since
a dead object's length is 0 and never wins a top-N ranking — not by design,
so it isn't a substitute fix.
"""

from __future__ import annotations

import numpy as np

from zarr_vectors_tools.multiresolution.object_selection import (
    apply_sparsity,
    select_by_length,
)


class TestAliveMask:
    def test_random_without_alive_mask_wastes_budget_on_dead_objects(self) -> None:
        # Half the population is already "dead" (as if dropped by an
        # earlier level).  Without alive_mask, a uniform draw over the
        # full 1000-object space only lands on a live object ~half the
        # time, even though 500 are requested.
        n = 1000
        rng = np.random.default_rng(0)
        alive = np.zeros(n, dtype=bool)
        alive[rng.choice(n, size=500, replace=False)] = True

        kept = apply_sparsity(n, 0.5, "random", seed=1)
        live_kept = np.count_nonzero(alive[kept])
        assert len(kept) == 500
        # Expect roughly half to land on a live object -- nowhere near 500.
        assert live_kept < 350

    def test_random_with_alive_mask_only_returns_live_objects(self) -> None:
        n = 1000
        rng = np.random.default_rng(0)
        alive = np.zeros(n, dtype=bool)
        alive[rng.choice(n, size=500, replace=False)] = True

        kept = apply_sparsity(n, 0.5, "random", seed=1, alive_mask=alive)
        assert len(kept) == 500
        assert np.all(alive[kept])

    def test_random_alive_mask_caps_target_below_alive_count(self) -> None:
        # Requesting more survivors than currently-alive objects should
        # clamp to the alive count rather than error or include dead ones.
        n = 1000
        alive = np.zeros(n, dtype=bool)
        alive[:50] = True

        kept = apply_sparsity(n, 0.5, "random", seed=1, alive_mask=alive)
        assert len(kept) == 50
        assert np.all(alive[kept])

    def test_length_strategy_also_restricted_to_alive_when_target_exceeds_it(
        self,
    ) -> None:
        # "length" is normally protected by ranking (dead objects have
        # length 0), but if target_count exceeds the alive count it must
        # still cap there rather than pad with dead (zero-length) objects.
        n = 100
        alive = np.zeros(n, dtype=bool)
        alive[:10] = True
        lengths = np.where(alive, 5.0, 0.0)

        kept = apply_sparsity(
            n, 0.5, "length", lengths=lengths, alive_mask=alive,
        )
        assert len(kept) == 10
        assert np.all(alive[kept])

    def test_cascading_random_sparsity_matches_absolute_targets_relative_to_original(
        self,
    ) -> None:
        # Default mode: each factor is an ABSOLUTE fraction of the original
        # count.  With increasing factors [1, 2, 8, 64, 512] every level's
        # kept count lands on round(n / factor) — the target never exceeds
        # the shrinking alive pool, so it doesn't collapse.  This guards the
        # relative_to="original" (default) semantics for direct callers.
        n = 100_000
        factors = [1, 2, 8, 64, 512]
        alive = np.ones(n, dtype=bool)
        for i, factor in enumerate(factors):
            kept = apply_sparsity(
                n, 1.0 / factor, "random", seed=i, alive_mask=alive,
            )
            expected = round(n / factor)
            assert len(kept) == expected, f"level {i + 1}: {len(kept)} != {expected}"
            new_alive = np.zeros(n, dtype=bool)
            new_alive[kept] = True
            alive = new_alive

    def test_cascading_random_sparsity_cumulative_relative_to_alive(self) -> None:
        # Cumulative mode (what pyramids use): a CONSTANT factor of 10 applied
        # five times keeps 1/10 of the SURVIVING pool each level -> geometric
        # decay 10000, 1000, 100, 10, 1.  The old code (relative_to="original")
        # returned 10000 EVERY level because target was round(n * 0.1) against
        # the constant original n; this is the regression guard for the
        # "pyramid levels 2..N are identical copies of level 1" bug.
        n = 100_000
        alive = np.ones(n, dtype=bool)
        expected = [10_000, 1_000, 100, 10, 1]
        for i, want in enumerate(expected):
            kept = apply_sparsity(
                n, 1.0 / 10.0, "random", seed=i, alive_mask=alive,
                relative_to="alive",
            )
            assert len(kept) == want, f"level {i + 1}: {len(kept)} != {want}"
            new_alive = np.zeros(n, dtype=bool)
            new_alive[kept] = True
            alive = new_alive

    def test_relative_to_alive_is_noop_without_alive_mask(self) -> None:
        # With no alive_mask every index is a candidate, so "alive" and
        # "original" bases coincide (len(candidates) == n_objects).
        n = 1000
        a = apply_sparsity(n, 0.25, "random", seed=1, relative_to="original")
        b = apply_sparsity(n, 0.25, "random", seed=1, relative_to="alive")
        assert len(a) == len(b) == 250
        assert np.array_equal(a, b)


class TestLengthStrategy:
    """`select_by_length` itself had no direct unit test anywhere before
    this — every existing coverage of "length" only exercised it
    incidentally, through `apply_sparsity`'s `alive_mask`/`relative_to`
    plumbing (see `TestAliveMask` above). See open-items.md item 11: a
    caller (`coarsen_skeleton_level`) passed FRAGMENT COUNT as `lengths` for
    over a year, and nothing here would have caught it, because nothing
    ever checked what `select_by_length` actually does with its input.
    """

    def test_keeps_the_top_n_by_whatever_is_in_lengths(self) -> None:
        # The function has no way to know what its input array MEANS -- it
        # only ranks it. This is the literal contract: top target_count by
        # value, nothing more.
        lengths = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        kept = select_by_length(lengths, target_count=3)
        assert set(kept.tolist()) == {5, 7, 4}  # values 9, 6, 5

    def test_a_shorter_object_can_lose_to_a_longer_one_even_amid_noise(self) -> None:
        # The property that was actually violated: given genuine lengths,
        # a strictly SHORTER object must never be kept over a strictly
        # LONGER one when there is room for only one of them.
        lengths = np.array([10.0, 50.0, 20.0])
        kept = select_by_length(lengths, target_count=1)
        assert kept.tolist() == [1]  # index of the 50.0, not 10.0 or 20.0

    def test_keeps_everything_when_target_meets_or_exceeds_the_population(self) -> None:
        lengths = np.array([1.0, 2.0, 3.0])
        kept = select_by_length(lengths, target_count=3)
        assert set(kept.tolist()) == {0, 1, 2}

    def test_apply_sparsity_length_forwards_the_real_lengths_array(self) -> None:
        # End-to-end through the dispatcher: the object apply_sparsity keeps
        # must be the same one select_by_length would pick directly on the
        # same array -- no silent substitution in between.
        n = 10
        lengths = np.array([5.0, 1.0, 9.0, 2.0, 7.0, 3.0, 8.0, 4.0, 6.0, 0.5])
        kept = apply_sparsity(n, 0.3, "length", lengths=lengths)
        direct = select_by_length(lengths, target_count=3)
        assert set(kept.tolist()) == set(direct.tolist()) == {2, 6, 4}  # 9, 8, 7

    def test_result_is_sorted_ascending(self) -> None:
        lengths = np.array([5.0, 9.0, 1.0, 7.0])
        kept = select_by_length(lengths, target_count=2)
        assert list(kept) == sorted(kept)
