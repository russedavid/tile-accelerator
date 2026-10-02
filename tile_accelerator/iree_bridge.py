"""Run target binaries through an IREE VM native-module boundary.

This host bridge uses the independent C++ simulator. It is not a custom HAL
hardware driver, and subprocess/file overhead is outside modeled chip timing.
"""

import numpy as np
from .isa import decode
from .runtime import execute

SOURCE = """module @tile_main {
  func.func private @tile.execute(!util.buffer, !util.buffer) -> !util.buffer
  func.func public @run(%program: !util.buffer, %input: !util.buffer) -> !util.buffer {
    %output = call @tile.execute(%program, %input) : (!util.buffer, !util.buffer) -> !util.buffer
    return %output : !util.buffer
  }
}
"""


class IreeBridge:
    def __init__(self, simulator):
        import iree.compiler as compiler
        import iree.runtime as rt

        self.rt = rt
        self.simulator = simulator
        binary = compiler.compile_str(SOURCE, target_backends=["llvm-cpu"])
        self.binary = binary
        simulator_path = simulator

        class State:
            def __init__(self, iface):
                self.last_counters = None

            def run(self, program_ref, input_ref):
                program = decode(bytes(program_ref.deref(rt.VmBuffer)))
                data = np.frombuffer(bytes(input_ref.deref(rt.VmBuffer)), dtype="<f4")
                result, counters = execute(program, data, simulator_path)
                self.last_counters = counters
                buffer = rt.VmBuffer(result.nbytes)
                memoryview(buffer)[:] = result.astype("<f4").tobytes()
                return buffer.ref

        interface = rt.PyModuleInterface("tile", State)
        interface.export("execute", "0rr_r", State.run)
        config = rt.Config("local-sync")
        native = interface.create()
        main = rt.VmModule.copy_buffer(config.vm_instance, binary)
        self.context = rt.SystemContext(
            vm_modules=(*config.default_vm_modules, native, main), config=config
        )
        self.entry = main.lookup_function("run")

    def __call__(self, program, data):
        rt = self.rt
        raw = program.encode()
        input_bytes = np.asarray(data, dtype="<f4").tobytes()
        a = rt.VmBuffer(len(raw))
        memoryview(a)[:] = raw
        b = rt.VmBuffer(len(input_bytes))
        memoryview(b)[:] = input_bytes
        arguments = rt.VmVariantList(2)
        arguments.push_ref(a)
        arguments.push_ref(b)
        results = rt.VmVariantList(1)
        self.context.vm_context.invoke(self.entry, arguments, results)
        output = results.get_as_object(0, rt.VmBuffer)
        return np.frombuffer(bytes(output), dtype="<f4").copy()
