import numpy as np
import pytest
from tile_accelerator.analysis import Architecture, analyze
from tile_accelerator.isa import Instruction as I, Op, Program


def test_known_dma_time_and_capacity():
    p = Program(
        [
            I(Op.LOAD2D, a=0, b=0, m=1, n=4, stride=4, event=1),
            I(Op.HALT, event=2, depends=1),
        ],
        np.zeros(4, np.float32),
        4,
        4,
        0,
        4,
    )
    result = analyze(p, Architecture(dma_bytes_per_cycle=8, dma_setup_cycles=2))
    assert result["cycles"] == 5
    assert result["dma_bytes"] == 16
    with pytest.raises(ValueError):
        analyze(p, Architecture(scratch_bytes=4))


def test_event_overlap_is_not_invented_when_dependency_serializes():
    serial = Program(
        [
            I(Op.LOAD2D, m=1, n=4, event=1),
            I(Op.ZERO, m=32, event=2, depends=1),
            I(Op.HALT, event=3, depends=2),
        ],
        np.zeros(4, np.float32),
        32,
        4,
        0,
        4,
    )
    independent = Program(
        [
            I(Op.LOAD2D, m=1, n=4, event=1),
            I(Op.ZERO, a=16, m=16, event=2),
            I(Op.HALT, event=3, depends=1),
        ],
        np.zeros(4, np.float32),
        32,
        4,
        0,
        4,
    )
    assert analyze(serial)["cycles"] > analyze(independent)["cycles"]


def test_optimistic_overlap_with_scratch_hazard_is_rejected():
    p=Program([I(Op.LOAD2D,a=0,b=0,m=1,n=4,stride=4,event=1),
               I(Op.ZERO,a=0,m=4,event=2),I(Op.HALT,event=3,depends=1)],np.zeros(4,np.float32),4,4,0,4)
    with pytest.raises(ValueError,match='scratch hazard'):analyze(p)
