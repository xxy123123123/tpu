"""Build the TPU SoC co-test firmware: rv32i bare-metal ELF -> BIN -> DAT.

DAT format: one 32-bit hex word per line, little-endian byte order, sequential
from the link base 0x4000_0000 (DDR model offset 0). Consumed by
sim/tpu_soc/tb_tpu_soc.v via $readmemh.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOOLS = ROOT / "xpack-riscv-none-elf-gcc-15.2.0-1/bin"
BUILD = HERE / "build"

FLAGS = [
    "-march=rv32i", "-mabi=ilp32", "-O2", "-std=c11", "-g",
    "-Wall", "-Wextra", "-Werror", "-ffreestanding", "-fno-builtin",
    "-fno-pic", "-fno-pie", "-msmall-data-limit=0",
    "-ffunction-sections", "-fdata-sections",
    "-nostdlib", "-nostartfiles",
    "-Wl,--no-relax", "-Wl,--build-id=none", "-Wl,--gc-sections",
]


def run(name, args):
    cmd = [str(TOOLS / ("riscv-none-elf-" + name + ".exe")), *map(str, args)]
    r = subprocess.run(cmd, cwd=BUILD, capture_output=True, text=True, timeout=60)
    (BUILD / (name + ".log")).write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode:
        raise RuntimeError(f"{name} failed:\n{r.stdout}{r.stderr}")


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    elf = BUILD / "program.elf"
    binary = BUILD / "program.bin"
    run("gcc", FLAGS + [HERE / "startup.S", HERE / "main.c",
                        "-T", HERE / "linker.ld",
                        f"-Wl,-Map={BUILD}/program.map", "-o", elf, "-lgcc"])
    run("objcopy", ["-O", "binary", elf, binary])
    run("size", [elf])
    raw = binary.read_bytes()
    raw += bytes((-len(raw)) % 4)
    (BUILD / "program.dat").write_text(
        "".join(f"{int.from_bytes(raw[i:i+4], 'little'):08x}\n"
                for i in range(0, len(raw), 4)))
    print(json.dumps({"status": "BUILD_PASS", "bytes": len(raw)},
                     indent=2))


if __name__ == "__main__":
    sys.exit(main())
