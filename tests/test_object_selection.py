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

from zarr_vectors_tools.multiresolution.object_selection import apply_sparsity


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

    def test_cascading_random_sparsity_matches_absolute_targets(self) -> None:
        # End-to-end: simulates the exact bug report -- 100_000 objects,
        # absolute (non-cascading) sparsity factors [1, 2, 8, 64, 512]
        # applied level-by-level with "random", each level's alive_mask
        # derived from the previous level's survivors.  Every level's
        # kept count should land on its absolute target, not collapse.
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
