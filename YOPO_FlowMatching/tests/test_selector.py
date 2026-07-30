#!/usr/bin/env python3
"""Small deterministic checks for continuity-aware lattice selection."""

import numpy as np

from yopo_flow.selection import ContinuityAwareSelector


def main():
    selector = ContinuityAwareSelector(endpoint_weight=0.1, hysteresis_margin=0.05)
    endpoints = np.zeros((3, 9), dtype=np.float32)
    endpoints[:, 1] = [-2.0, 0.0, 2.0]
    first = selector.select(np.asarray([0.1, 0.2, 0.3]), endpoints)
    assert first.index == 0
    # A tiny score advantage on the opposite side must not trigger a switch.
    second = selector.select(np.asarray([0.12, 0.2, 0.10]), endpoints)
    assert second.index == 0
    # A large score improvement releases the hysteresis.
    third = selector.select(np.asarray([1.0, 0.5, 0.0]), endpoints)
    assert third.index == 2
    assert third.switched
    print("SELECTOR TEST PASSED", selector.switch_rate, selector.max_endpoint_jump)


if __name__ == "__main__":
    main()
