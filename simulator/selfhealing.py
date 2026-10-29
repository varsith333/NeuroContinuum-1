"""Self-healing synapse repair — turning a brittle crossbar into a resilient one.

Silicon ages, radiates, and develops stuck-at faults; analog/memristive synapses
drift and get pinned to a rail.  A dense accelerator can shrug off one dead
cell, but a *learning* neuromytic core whose plastic synapses fail will silently
degrade.  NeuroContinuum-1 watches its own synapses and repairs them at runtime.

This module is the software golden model for ``rtl/fault_monitor.v`` +
``rtl/synapse_remap.v``.

Crossbar repair model (must match the hardware)
------------------------------------------------
The weight matrix is ``(n_rows x n_inputs)`` functional cells plus ``n_spare``
**extra columns** that connect to no input at reset.

* Physical storage is ``(n_rows x (n_inputs + n_spare))``.
* ``_map[i][j]`` = physical column currently serving synapse ``(i, j)``
  (identity at reset: ``_map[i][j] == j``).
* ``live_view()`` is always ``(n_rows, n_inputs)`` -> ``W_phys[i, _map[i][j]]``,
  so the network always sees a full-width weight matrix whether or not a repair
  happened (repairs are transparent to the network).
* A spare column ``s`` is one shared wire; claiming it for input ``j`` makes it a
  *mirror of input j* (``_col_src[s] = j``) used by exactly one row.  A later
  repair may reuse ``s`` only for the same input ``j``.

Fault detection = write-refusal (no false positives)
----------------------------------------------------
Each step, the plasticity rule produces its *desired* weights; we write them to
the physical cells; then we read back and compare.  A healthy cell always shows
``actual == desired``.  A **stuck** cell rejects the write, so
``|actual - desired| > delta_eps`` *and* the rule had wanted to move it.  We
require this write-refusal to persist for ``stuck_cycles`` confirmations (a real
monitor debounces transient glitches) before repairing — so a synapse that is
legitimately pinned at a rail because the rule *wants* it there is never
mistaken for a fault (that is the classic false-positive trap).

Repair = re-point ``_map[i][j]`` at a free spare column.  If none can mirror
input ``j``, the row is marked **degraded** and homeostatic scaling is tightened
instead — graceful degradation, never a crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FaultReport:
    suspect: list = field(default_factory=list)      # [(row, input_col), ...]
    remapped: list = field(default_factory=list)     # [(row, col, phys_spare), ...]
    degraded_rows: set = field(default_factory=set)  # rows that ran out of spares
    spare_usage: dict = field(default_factory=dict)  # {row: n_spares_used}
    n_repairs: int = 0


class SelfHealingSynapseArray:
    """Wraps a ``(n_rows, n_inputs)`` weight matrix with fault detection + repair."""

    def __init__(self, W: np.ndarray, spare_frac: float = 0.15,
                 max_spares: int = 8, stuck_cycles: int = 3,
                 delta_eps: float = 1e-4, w_max: float = 1.0):
        W = np.array(W, dtype=float, copy=True)
        if W.ndim != 2:
            raise ValueError("W must be 2-D (n_rows, n_inputs)")
        self.n_rows, self.n_inputs = W.shape
        self.n_spare = int(min(max_spares, max(1, round(self.n_inputs * spare_frac))))
        self.stuck_cycles = int(stuck_cycles)
        self.delta_eps = float(delta_eps)
        self.w_max = float(w_max)

        n_phys = self.n_inputs + self.n_spare
        self.W_phys = np.zeros((self.n_rows, n_phys))
        self.W_phys[:, :self.n_inputs] = W
        self._map = np.tile(np.arange(self.n_inputs), (self.n_rows, 1))
        self._col_src = np.full(self.n_spare, -1, dtype=int)  # spare -> input mirrored
        self._col_owner = np.full(self.n_spare, -1, dtype=int)

        # stuck-cell model: mask + the value a stuck cell refuses to change from
        self._stuck = np.zeros((self.n_rows, n_phys), dtype=bool)
        self._pin = np.zeros((self.n_rows, n_phys), dtype=float)
        self._refusal = np.zeros((self.n_rows, self.n_inputs), dtype=int)
        self.report = FaultReport()

    # ------------------------------------------------------------------ views
    def live_view(self) -> np.ndarray:
        """(n_rows, n_inputs) matrix the network actually uses this step."""
        return np.take_along_axis(self.W_phys, self._map, axis=1)

    # ---------------------------------------------------------------- plasticity
    def update(self, pre: np.ndarray, post: np.ndarray, rule_step) -> np.ndarray:
        """Run one plasticity step, write back, then screen for write-refusals.

        ``rule_step(W, pre, post)`` mutates ``W`` in place (BCM, STDP, ...).
        """
        before = self.live_view().copy()
        desired = before.copy()
        rule_step(desired, np.asarray(pre), np.asarray(post))

        np.put_along_axis(self.W_phys, self._map, desired, axis=1)
        # a stuck physical cell rejects the write: it keeps its pinned value
        if self._stuck.any():
            self.W_phys = np.where(self._stuck, self._pin, self.W_phys)

        actual = self.live_view()
        self._screen(before, desired, actual)
        return actual

    def set_live(self, M: np.ndarray) -> None:
        """Overwrite the live weights (used by homeostatic rescaling etc.)."""
        M = np.asarray(M, dtype=float)
        if M.shape != (self.n_rows, self.n_inputs):
            raise ValueError(f"shape {M.shape} != {(self.n_rows, self.n_inputs)}")
        np.put_along_axis(self.W_phys, self._map, M, axis=1)
        if self._stuck.any():
            self.W_phys = np.where(self._stuck, self._pin, self.W_phys)

    # -------------------------------------------------------------- fault inject
    def inject_fault(self, row: int, col: int, mode: str = "stuck") -> None:
        """Test hook: make the physical cell serving (row, input col) refuse writes."""
        phys = self._map[row, col]
        cur = float(self.W_phys[row, phys])
        self._stuck[row, phys] = True
        if mode == "stuck_high":
            self._pin[row, phys] = self.w_max
        elif mode == "stuck_low":
            self._pin[row, phys] = 0.0
        elif mode == "stuck":                 # frozen at its current value
            self._pin[row, phys] = cur
        else:
            raise ValueError(mode)

    # ---------------------------------------------------------------- screening
    def _screen(self, before: np.ndarray, desired: np.ndarray,
                actual: np.ndarray) -> None:
        wanted = np.abs(desired - before) > self.delta_eps   # rule wanted to move
        refused = np.abs(actual - desired) > self.delta_eps  # write didn't take
        stuck_now = wanted & refused
        # debounce: require consecutive refusals before acting
        self._refusal = np.where(stuck_now, self._refusal + 1, 0)
        for r, c in np.argwhere(self._refusal >= self.stuck_cycles):
            self.report.suspect.append((int(r), int(c)))
            self._repair(int(r), int(c))

    # ------------------------------------------------------------------ repair
    def _repair(self, row: int, col: int) -> bool:
        j = col                                   # input index this synapse hears
        old_phys = self._map[row, col]

        choice = -1
        # prefer a spare already mirroring input j, else a free spare
        for pool in (np.flatnonzero(self._col_src == j),
                     np.flatnonzero(self._col_src == -1)):
            if pool.size:
                choice = int(pool[0]); break

        if choice == -1:
            self.report.degraded_rows.add(row)
            self._refusal[row, col] = 0
            return False

        phys_spare = self.n_inputs + choice
        self.W_phys[row, phys_spare] = self.W_phys[row, old_phys]  # carry value
        if self._col_src[choice] == -1:
            self._col_src[choice] = j
        self._col_owner[choice] = row
        self._map[row, col] = phys_spare

        self.report.remapped.append((row, col, phys_spare))
        self.report.n_repairs += 1
        self.report.spare_usage[row] = self.report.spare_usage.get(row, 0) + 1
        self._refusal[row, col] = 0
        return True

    # -------------------------------------------------------------- introspection
    @property
    def spares_remaining(self) -> int:
        return int(np.sum(self._col_src == -1))

    def summary(self) -> str:
        r = self.report
        return (f"repairs={r.n_repairs} suspect={len(r.suspect)} "
                f"remapped={len(r.remapped)} spares_left={self.spares_remaining} "
                f"degraded_rows={sorted(r.degraded_rows)}")
