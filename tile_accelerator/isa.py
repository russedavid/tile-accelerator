"""Little-endian v1 binary format, shared by compiler and C++ execution engine."""

from dataclasses import dataclass
from enum import IntEnum
import struct
import numpy as np

MAGIC = b"TILEAC01"
VERSION = 1
HEADER = struct.Struct("<8s7I")
INSTRUCTION = struct.Struct("<16I")
MAX_ELEMENTS = 16_000_000
MAX_INSTRUCTIONS = 1_000_000


class Op(IntEnum):
    LOAD2D = 1
    STORE2D = 2
    ZERO = 3
    MATMUL = 4
    EPILOGUE = 5
    WAIT = 6
    DEPTHWISE = 7
    HALT = 8


@dataclass(frozen=True)
class Instruction:
    op: Op
    a: int = 0
    b: int = 0
    c: int = 0
    m: int = 0
    n: int = 0
    k: int = 0
    flags: int = 0
    event: int = 0
    depends: int = 0
    stride: int = 0
    aux: int = 0
    p0: int = 0
    p1: int = 0
    p2: int = 0
    p3: int = 0

    def encode(self):
        values = [int(getattr(self, n)) for n in self.__dataclass_fields__]
        if any(v < 0 or v > 0xFFFFFFFF for v in values):
            raise ValueError("instruction field outside uint32")
        return INSTRUCTION.pack(*values)


@dataclass
class Program:
    instructions: list[Instruction]
    memory: np.ndarray
    scratch_elements: int
    input_elements: int
    output_offset: int
    output_elements: int

    def encode(self):
        if not self.instructions or len(self.instructions) > MAX_INSTRUCTIONS:
            raise ValueError("invalid instruction count")
        if self.instructions[-1].op != Op.HALT:
            raise ValueError("program must end in HALT")
        mem = np.asarray(self.memory, dtype="<f4").reshape(-1)
        if (
            not 0 < mem.size <= MAX_ELEMENTS
            or not 0 < self.scratch_elements <= MAX_ELEMENTS
        ):
            raise ValueError("invalid memory capacity")
        if not 0 < self.input_elements <= mem.size:
            raise ValueError("invalid input extent")
        if (
            not 0 < self.output_elements
            or self.output_offset + self.output_elements > mem.size
        ):
            raise ValueError("invalid output extent")
        return (
            HEADER.pack(
                MAGIC,
                VERSION,
                len(self.instructions),
                mem.size,
                self.scratch_elements,
                self.input_elements,
                self.output_offset,
                self.output_elements,
            )
            + b"".join(i.encode() for i in self.instructions)
            + mem.tobytes()
        )


def decode(blob):
    if len(blob) < HEADER.size:
        raise ValueError("truncated header")
    magic, version, count, mem, scratch, inp, out, size = HEADER.unpack_from(blob)
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported binary format")
    if (
        not 0 < count <= MAX_INSTRUCTIONS
        or not 0 < mem <= MAX_ELEMENTS
        or not 0 < scratch <= MAX_ELEMENTS
    ):
        raise ValueError("invalid binary resource limits")
    if not 0 < inp <= mem or not 0 < size or out + size > mem:
        raise ValueError("invalid binary IO extents")
    if len(blob) != HEADER.size + count * INSTRUCTION.size + mem * 4:
        raise ValueError("binary length mismatch")
    instructions = []
    for ix in range(count):
        values = INSTRUCTION.unpack_from(blob, HEADER.size + ix * INSTRUCTION.size)
        instructions.append(Instruction(Op(values[0]), *values[1:]))
    if instructions[-1].op != Op.HALT:
        raise ValueError("program must end in HALT")
    memory = np.frombuffer(
        blob, dtype="<f4", offset=HEADER.size + count * INSTRUCTION.size
    ).copy()
    return Program(instructions, memory, scratch, inp, out, size)


def disassemble(program):
    lines = [
        f"TILEAC v{VERSION} memory={program.memory.size} scratch={program.scratch_elements} input={program.input_elements} output={program.output_offset}+{program.output_elements}"
    ]
    for ix, i in enumerate(program.instructions):
        fields = " ".join(f"{k}={v}" for k, v in i.__dict__.items() if k != "op" and v)
        lines.append(f"{ix:06d} {i.op.name} {fields}")
    return "\n".join(lines) + "\n"
