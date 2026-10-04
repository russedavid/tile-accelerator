"""Resource timeline and action-count estimates, not silicon measurements."""

from dataclasses import dataclass, asdict
import math
from .isa import Op


@dataclass(frozen=True)
class Architecture:
    dma_bytes_per_cycle: float = 16
    macs_per_cycle: float = 64
    vector_elements_per_cycle: float = 16
    clock_hz: float = 200_000_000
    dma_setup_cycles: int = 12
    dram_pj_per_byte: float = 20
    scratch_pj_per_byte: float = 1
    mac_pj: float = 4
    leakage_mw: float = 1
    scratch_bytes: int = 16384

    def validate(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("finite architecture parameters required")
        if (
            min(
                self.dma_bytes_per_cycle,
                self.macs_per_cycle,
                self.vector_elements_per_cycle,
                self.clock_hz,
            )
            <= 0
        ):
            raise ValueError("positive rates required")
        if (
            min(
                self.dma_setup_cycles,
                self.dram_pj_per_byte,
                self.scratch_pj_per_byte,
                self.mac_pj,
                self.leakage_mw,
            )
            < 0
            or self.scratch_bytes < 4
        ):
            raise ValueError("negative cost or invalid capacity")


def validate_scratch_timeline(program, trace):
    """Reject modeled overlap with a scratch RAW/WAR/WAW conflict."""
    intervals=[]
    for instruction,item in zip(program.instructions,trace):
        i=instruction;accesses=[]
        if i.op==Op.LOAD2D:accesses=[(i.b,i.m*i.n,True)]
        elif i.op==Op.STORE2D:accesses=[(i.a,i.m*i.n,False)]
        elif i.op==Op.ZERO:accesses=[(i.a,i.m,True)]
        elif i.op==Op.MATMUL:accesses=[(i.a,i.m*i.k,False),(i.b,i.k*i.n,False),(i.c,i.m*i.n,True)]
        elif i.op==Op.EPILOGUE:
            accesses=[(i.a,i.m*i.n,True),(i.b,i.n,False)]
            if i.flags&2:accesses.append((i.c,i.m*i.n,False))
        for address,size,write in accesses:
            if size:intervals.append((item["start_cycle"],item["end_cycle"],address,address+size,write,item["pc"]))
    active=[]
    for current in sorted(intervals,key=lambda item:item[0]):
        start,end,left,right,write,pc=current
        active=[item for item in active if item[1]>start]
        for other in active:
            if pc!=other[5] and (write or other[4]) and left<other[3] and other[2]<right:
                raise ValueError(f"scratch hazard between instructions {other[5]} and {pc}")
        active.append(current)


def analyze(program, architecture=Architecture()):
    architecture.validate()
    if program.scratch_elements * 4 > architecture.scratch_bytes:
        raise ValueError("program scratch requirement exceeds architecture")
    finish = {0: 0}
    available = {"dma": 0, "matrix": 0, "vector": 0, "control": 0}
    trace = []
    dma_bytes = 0
    macs = 0
    scratch_bytes = 0
    for pc, i in enumerate(program.instructions):
        if i.event != pc + 1 or i.depends not in finish:
            raise ValueError("invalid dependency")
        resource = "control"
        duration = 1
        byte_count = 0
        ops = 0
        if i.op in (Op.LOAD2D, Op.STORE2D):
            resource = "dma"
            byte_count = i.m * i.n * 4
            duration = architecture.dma_setup_cycles + math.ceil(
                byte_count / architecture.dma_bytes_per_cycle
            )
            dma_bytes += byte_count
            scratch_bytes += byte_count
        elif i.op == Op.MATMUL:
            resource = "matrix"
            ops = i.m * i.n * i.k
            duration = math.ceil(ops / architecture.macs_per_cycle)
            macs += ops
            scratch_bytes += (i.m * i.k + i.k * i.n + 2 * i.m * i.n) * 4
        elif i.op == Op.EPILOGUE:
            resource = "vector"
            duration = math.ceil(i.m * i.n / architecture.vector_elements_per_cycle)
            scratch_bytes += (
                2 * i.m * i.n + i.n + (i.m * i.n if i.flags & 2 else 0)
            ) * 4
        elif i.op == Op.ZERO:
            resource = "vector"
            duration = math.ceil(i.m / architecture.vector_elements_per_cycle)
            scratch_bytes += i.m * 4
        elif i.op == Op.DEPTHWISE:
            resource = "matrix"
            ops = i.m * i.n * i.k * 9
            # Upper bound counts padding positions as work. Native unit timing
            # includes direct global traffic; no unvalidated cache benefit.
            byte_count = (i.m * i.n * i.k * 10 + i.k * 9 + i.k) * 4
            duration = math.ceil(ops / architecture.macs_per_cycle) + math.ceil(
                byte_count / architecture.dma_bytes_per_cycle
            )
            macs += ops
            dma_bytes += byte_count
        elif i.op not in (Op.WAIT, Op.HALT):
            raise ValueError("unknown instruction")
        if i.op == Op.MATMUL and i.aux not in finish:
            raise ValueError("invalid accumulator dependency")
        second = finish[i.aux] if i.op == Op.MATMUL else 0
        start = max(finish[i.depends], second, available[resource])
        end = start + max(1, duration)
        available[resource] = end
        finish[i.event] = end
        trace.append(
            {
                "pc": pc,
                "op": i.op.name,
                "resource": resource,
                "start_cycle": start,
                "end_cycle": end,
                "depends": i.depends,
                "dma_bytes": byte_count,
                "macs": ops,
            }
        )
    validate_scratch_timeline(program, trace)
    cycles = max(finish.values())
    seconds = cycles / architecture.clock_hz
    dynamic_pj = (
        dma_bytes * architecture.dram_pj_per_byte
        + scratch_bytes * architecture.scratch_pj_per_byte
        + macs * architecture.mac_pj
    )
    leakage_j = architecture.leakage_mw / 1000 * seconds
    return {
        "kind": "analytical estimate; not cycle-accurate or measured silicon",
        "cycles": cycles,
        "latency_us": seconds * 1e6,
        "dma_bytes": dma_bytes,
        "scratch_action_bytes": scratch_bytes,
        "multiply_adds": macs,
        "dynamic_energy_uj": dynamic_pj / 1e6,
        "leakage_energy_uj": leakage_j * 1e6,
        "energy_uj": dynamic_pj / 1e6 + leakage_j * 1e6,
        "architecture": asdict(architecture),
        "trace": trace,
        "assumptions": [
            "one DMA engine and one matrix engine",
            "event-defined dependencies",
            "no caches, arbitration, bank conflicts or pipeline model",
            "user-supplied component energy and clock; estimates need calibration",
        ],
    }


def sweep(shape, weights):
    from .compiler import Target, compile_mlp

    points = []
    for scratch in [8192, 16384, 32768]:
        for tile_m in [8, 16, 32]:
            for tile_n in [8, 16, 32]:
                for mac_rate in [32, 64, 128]:
                    target = Target(
                        scratch_bytes=scratch, tile_m=tile_m, tile_n=tile_n, tile_k=16
                    )
                    try:
                        program, _ = compile_mlp(shape, weights, target)
                    except ValueError:
                        continue
                    estimate = analyze(
                        program,
                        Architecture(scratch_bytes=scratch, macs_per_cycle=mac_rate),
                    )
                    points.append(
                        {
                            "scratch_bytes": scratch,
                            "tile_m": tile_m,
                            "tile_n": tile_n,
                            "macs_per_cycle": mac_rate,
                            "latency_us": estimate["latency_us"],
                            "energy_uj": estimate["energy_uj"],
                            "area_proxy": scratch / 8192 + mac_rate / 32,
                        }
                    )
    for p in points:
        p["pareto"] = not any(
            all(q[k] <= p[k] for k in ["latency_us", "energy_uj", "area_proxy"])
            and any(q[k] < p[k] for k in ["latency_us", "energy_uj", "area_proxy"])
            for q in points
        )
    return {
        "kind": "architecture/tile sensitivity study; illustrative area proxy, not ASIC area",
        "shape": list(shape),
        "points": points,
    }
